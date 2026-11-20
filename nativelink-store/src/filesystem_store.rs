// Copyright 2024 Trace Machina, Inc. All rights reserved.
//
// Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    See LICENSE file for details
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use core::cmp;
use core::fmt::{Debug, Display, Formatter};
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use std::borrow::Cow;
#[cfg(unix)]
use std::collections::HashMap;
use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::io::SeekFrom;
use std::sync::{Arc, Weak};
use std::time::SystemTime;

use async_lock::RwLock;
#[cfg(unix)]
use async_lock::{Mutex, MutexGuardArc};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::stream::{StreamExt, TryStreamExt};
use futures::{Future, TryFutureExt};
use nativelink_config::stores::FilesystemSpec;
use nativelink_error::{Code, Error, ResultExt, make_err};
use nativelink_metric::MetricsComponent;
use nativelink_util::buf_channel::{
    DropCloserReadHalf, DropCloserWriteHalf, make_buf_channel_pair,
};
use nativelink_util::common::{DigestInfo, fs};
use nativelink_util::evicting_map::{EvictingMap, EvictionSnapshot, InsertCleanup, LenEntry};
use nativelink_util::fs::FileSlot;
use nativelink_util::health_utils::{HealthRegistryBuilder, HealthStatus, HealthStatusIndicator};
use nativelink_util::spawn_blocking;
#[cfg(unix)]
use nativelink_util::store_trait::RemoveItemCallback;
use nativelink_util::store_trait::{
    RemoveCallback, StoreDriver, StoreKey, StoreKeyBorrow, StoreOptimizations, UploadSizeInfo,
};
use nativelink_util::{background_spawn, background_spawn_blocking};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, Take};
use tokio::sync::Semaphore;
use tokio::time::timeout;
use tokio_stream::wrappers::ReadDirStream;
use tracing::{debug, error, info, trace, warn};

use crate::callback_utils::RemoveCallbackHolder;
use crate::cas_utils::is_zero_digest;

// Default size to allocate memory of the buffer when reading files.
const DEFAULT_BUFF_SIZE: usize = 32 * 1024;
// Default block size of all major filesystems is 4KB
const DEFAULT_BLOCK_SIZE: u64 = 4 * 1024;

pub const STR_FOLDER_V1: &str = "s";
pub const DIGEST_FOLDER_V1: &str = "d";

pub const STR_FOLDER_V2: &str = "s2";
pub const DIGEST_FOLDER_V2: &str = "d2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Version {
    Flat,
    V1,
    V2,
}

/// Suffix for the sibling directory that holds per-digest read-only
/// **executable** (0o555) variants of CAS blobs (see
/// [`FilesystemStore::get_executable_hardlink_source`]). It is a sibling of
/// `content_path` rather than a child so the normal content/temp scan and prune
/// logic never touches it. Cleared on writable startup; entries are
/// regenerable. If the wipe is blocked by a read-only filesystem, executable
/// variants are disabled so surviving files are never trusted.
#[cfg(unix)]
const EXECUTABLE_DIR_SUFFIX: &str = ".exec";

const MAX_CONCURRENT_VARIANT_LOOKUPS: usize = 32;

#[derive(Clone, Copy, Debug)]
pub enum FileType {
    Digest,
    String,
}

#[derive(Debug, MetricsComponent)]
pub struct SharedContext {
    // Used in testing to know how many active drop() spawns are running.
    // TODO(palfrey) It is probably a good idea to use a spin lock during
    // destruction of the store to ensure that all files are actually
    // deleted (similar to how it is done in tests).
    #[metric(help = "Number of active drop spawns")]
    pub active_drop_spawns: AtomicU64,
    #[metric(help = "Path to the configured temp path")]
    temp_path: String,
    #[metric(help = "Path to the configured content path")]
    content_path: String,
    /// Files of retired entries waiting to be deleted. See
    /// [`SharedContext::retire_file`].
    retired_files: parking_lot::Mutex<RetiredFiles>,
    /// Admits one retirement operation of this store at a time. See
    /// [`SharedContext::retire_blocking`].
    retirement: Arc<Semaphore>,
}

/// The files of retired entries that are waiting to be deleted, and whether a
/// drain task is deleting them.
#[derive(Debug, Default)]
struct RetiredFiles {
    paths: VecDeque<OsString>,
    draining: bool,
}

impl SharedContext {
    fn new(temp_path: String, content_path: String) -> Self {
        Self {
            active_drop_spawns: AtomicU64::new(0),
            temp_path,
            content_path,
            retired_files: parking_lot::Mutex::default(),
            retirement: Arc::new(Semaphore::new(1)),
        }
    }

    /// Runs `operation`, a retirement operation of this store, on the
    /// blocking pool once no other one is running: the rename of an evicted
    /// entry's file into the temp directory ([`FileEntryImpl::unref`]) or the
    /// unlink of an evicted digest's executable variant
    /// ([`ExecutableVariantRemover`]).
    ///
    /// The kernel performs one rename between two directories at a time per
    /// filesystem, and one unlink at a time per directory. Evictions retire
    /// thousands of entries at once, and several cleanups can overlap (each
    /// write retires what its insertion displaced after releasing its write
    /// permit); admitted together, each waiting operation held a
    /// blocking-pool thread, until none was left for any other file
    /// operation in the process, while retiring nothing faster. Waiting here
    /// holds no thread. The admission moves into the blocking closure, so the
    /// next operation starts only when this one has really finished, even if
    /// the caller is dropped meanwhile; a closure dropped before it starts
    /// releases it unrun. It takes no open-file permit: a rename or an unlink
    /// opens no file, and waiting for one while admitted could hold every
    /// retirement of the store behind the writers whose files it retires.
    async fn retire_blocking<T: Send + 'static>(
        &self,
        operation: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T, Error> {
        let admission = self
            .retirement
            .clone()
            .acquire_owned()
            .await
            .map_err(|err| Error::from_std_err(Code::Internal, &err))?;
        spawn_blocking!("filesystem_store_retire", move || {
            let _admission = admission;
            operation()
        })
        .await
        .map_err(Error::from)
    }

    /// Deletes the file of a retired entry, in the background.
    ///
    /// Retired files are deleted one at a time, in the order they were
    /// retired, by at most one drain task per store. Nearly all of them are in
    /// the store's temp directory, which an eviction fills with one file per
    /// evicted entry (thousands at once), and the kernel performs only one
    /// unlink at a time in a directory: each holds the directory's inode lock.
    /// A task per file, as before, put one blocking-pool thread per file to
    /// sleep on that lock, until no thread was left for any other file
    /// operation in the process (reads, writes and fsyncs of every store and
    /// the worker), while deleting no faster than one thread does.
    fn retire_file(self: &Arc<Self>, file_path: OsString) {
        let mut retired_files = self.retired_files.lock();
        retired_files.paths.push_back(file_path);
        if retired_files.draining {
            // The running drain takes the file before it can find the queue
            // empty: both happen under this lock.
            return;
        }
        retired_files.draining = true;
        drop(retired_files);
        let drain = RetiredFileDrain {
            shared_context: self.clone(),
            finished: false,
        };
        background_spawn_blocking!("filesystem_delete_files", move || drain.run());
    }
}

/// The one task allowed to delete a store's retired files, while it exists.
/// Dropping it unfinished (its blocking closure was dropped without running
/// to the end, e.g. at runtime shutdown) lets the next retired file start a
/// new drain, which also deletes the files this one left queued.
struct RetiredFileDrain {
    shared_context: Arc<SharedContext>,
    finished: bool,
}

impl RetiredFileDrain {
    fn run(mut self) {
        loop {
            let file_path = {
                let mut retired_files = self.shared_context.retired_files.lock();
                let Some(file_path) = retired_files.paths.pop_front() else {
                    retired_files.draining = false;
                    self.finished = true;
                    return;
                };
                file_path
            };
            // Deliberately NOT `fs::remove_file`: unlinking opens no file, so
            // it needs no open-file permit, and waiting for one here could
            // hold the drain behind the writers whose files it is deleting.
            match std::fs::remove_file(&file_path) {
                Ok(()) => debug!(?file_path, "File deleted"),
                // The file already being gone is the desired end state of a
                // delete, not a failure — e.g. an entry marked Temp after an
                // already-gone unref points at a path that was never created.
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    debug!(?file_path, "File already gone, nothing to delete");
                }
                Err(err) => error!(?file_path, ?err, "Failed to delete file"),
            }
            // .fetch_sub returns previous value, so we subtract one to get approximate current value
            let current_active_drop_spawns = self
                .shared_context
                .active_drop_spawns
                .fetch_sub(1, Ordering::Relaxed)
                - 1;
            debug!(
                ?current_active_drop_spawns,
                ?file_path,
                "Dropped a filesystem_delete_file"
            );
        }
    }
}

impl Drop for RetiredFileDrain {
    fn drop(&mut self) {
        if !self.finished {
            self.shared_context.retired_files.lock().draining = false;
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
enum PathType {
    Content,
    Temp,
    Custom(OsString),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Generation(u64);

impl Generation {
    pub const fn new(generation: u64) -> Self {
        Self(generation)
    }

    pub const fn inner(&self) -> u64 {
        self.0
    }
}

impl Display for Generation {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// [`EncodedFilePath`] stores the path to the file
/// including the context, path type and key to the file.
/// The whole [`StoreKey`] is stored as opposed to solely
/// the [`DigestInfo`] so that it is more usable for things
/// such as BEP -see Issue #1108
#[derive(Debug)]
pub struct EncodedFilePath {
    shared_context: Arc<SharedContext>,
    path_type: PathType,
    key: StoreKey<'static>,
    generation: Generation,
    version: Version,
}

impl EncodedFilePath {
    #[inline]
    fn get_file_path(&self) -> Cow<'_, OsStr> {
        get_file_path_raw(
            &self.path_type,
            self.shared_context.as_ref(),
            &self.key,
            self.generation,
            self.version,
        )
    }
}

#[inline]
fn get_file_path_raw<'a>(
    path_type: &'a PathType,
    shared_context: &SharedContext,
    key: &StoreKey<'a>,
    generation: Generation,
    version: Version,
) -> Cow<'a, OsStr> {
    let folder = match path_type {
        PathType::Content => &shared_context.content_path,
        PathType::Temp => &shared_context.temp_path,
        PathType::Custom(path) => return Cow::Borrowed(path),
    };
    Cow::Owned(to_full_path_from_key(folder, key, generation, version))
}

impl Drop for EncodedFilePath {
    fn drop(&mut self) {
        // `drop()` can be called during shutdown, so we use `path_type` flag to know if the
        // file actually needs to be deleted.
        if self.path_type == PathType::Content {
            return;
        }

        let file_path = self.get_file_path().to_os_string();
        // .fetch_add returns previous value, so we add one to get approximate current value
        let current_active_drop_spawns = self
            .shared_context
            .active_drop_spawns
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        debug!(
            %current_active_drop_spawns,
            ?file_path,
            "Spawned a filesystem_delete_file"
        );
        self.shared_context.retire_file(file_path);
    }
}

/// This creates the file path from the [`StoreKey`]. If
/// it is a string, the string, prefixed with [`STR_PREFIX`]
/// for backwards compatibility, is stored.
///
/// If it is a [`DigestInfo`], it is prefixed by [`DIGEST_PREFIX`]
/// followed by the string representation of a digest - the hash in hex,
/// a hyphen then the size in bytes
///
/// Previously, only the string representation of the [`DigestInfo`] was
/// used with no prefix
#[inline]
fn to_full_path_from_key(
    folder: &str,
    key: &StoreKey<'_>,
    generation: Generation,
    version: Version,
) -> OsString {
    match (key, version) {
        (StoreKey::Digest(digest_info), Version::Flat) => format!("{folder}/{digest_info}"),
        (StoreKey::Str(str), Version::Flat) => format!("{folder}/{str}"),
        (StoreKey::Str(str), Version::V2) => {
            format!("{folder}/{STR_FOLDER_V2}/{str}-{generation}")
        }
        (StoreKey::Str(str), Version::V1) => format!("{folder}/{STR_FOLDER_V1}/{str}"),
        (StoreKey::Digest(digest_info), Version::V2) => {
            format!("{folder}/{DIGEST_FOLDER_V2}/{digest_info}-{generation}")
        }
        (StoreKey::Digest(digest_info), Version::V1) => {
            format!("{folder}/{DIGEST_FOLDER_V1}/{digest_info}")
        }
    }
    .into()
}

pub trait FileEntry: LenEntry + Send + Sync + Debug + 'static {
    /// Responsible for creating the underlying `FileEntry`.
    fn create(
        data_size: u64,
        block_size: u64,
        generation: Generation,
        encoded_file_path: RwLock<EncodedFilePath>,
    ) -> Self;

    /// Creates a (usually) temp file, opens it and returns the path to the temp file.
    fn make_and_open_file(
        block_size: u64,
        generation: Generation,
        encoded_file_path: EncodedFilePath,
    ) -> impl Future<Output = Result<(Self, FileSlot, OsString), Error>> + Send
    where
        Self: Sized;

    /// Returns the file generation.
    fn generation(&self) -> Generation;

    /// Returns the underlying size of the data in bytes
    fn data_size(&self) -> u64;

    /// Returns the underlying reference to the size of the data in bytes
    fn data_size_mut(&mut self) -> &mut u64;

    /// Returns the actual size of the underlying file on the disk after accounting for filesystem block size.
    fn size_on_disk(&self) -> u64;

    /// Gets the underlying `EncodedfilePath`.
    fn get_encoded_file_path(&self) -> &RwLock<EncodedFilePath>;

    /// Returns a reader that will read part of the underlying file.
    fn read_file_part(
        &self,
        offset: u64,
        length: u64,
    ) -> impl Future<Output = Result<Take<FileSlot>, Error>> + Send;

    /// This function is a safe way to extract the file name of the underlying file. To protect users from
    /// accidentally creating undefined behavior we encourage users to do the logic they need to do with
    /// the filename inside this function instead of extracting the filename and doing the logic outside.
    /// This is because the filename is not guaranteed to exist after this function returns, however inside
    /// the callback the file is always guaranteed to exist and immutable.
    /// DO NOT USE THIS FUNCTION TO EXTRACT THE FILENAME AND STORE IT FOR LATER USE.
    fn get_file_path_locked<
        T,
        Fut: Future<Output = Result<T, Error>> + Send,
        F: FnOnce(OsString) -> Fut + Send,
    >(
        &self,
        handler: F,
    ) -> impl Future<Output = Result<T, Error>> + Send;
}

pub struct FileEntryImpl {
    data_size: u64,
    block_size: u64,
    // We lock around this as it gets rewritten when we move between temp and content types
    encoded_file_path: RwLock<EncodedFilePath>,
    generation: Generation,
}

impl FileEntryImpl {
    pub fn get_shared_context_for_test(&mut self) -> Arc<SharedContext> {
        self.encoded_file_path.get_mut().shared_context.clone()
    }
}

impl FileEntry for FileEntryImpl {
    fn create(
        data_size: u64,
        block_size: u64,
        generation: Generation,
        encoded_file_path: RwLock<EncodedFilePath>,
    ) -> Self {
        Self {
            data_size,
            block_size,
            encoded_file_path,
            generation,
        }
    }

    /// This encapsulates the logic for the edge case of if the file fails to create
    /// the cleanup of the file is handled without creating a `FileEntry`, which would
    /// try to cleanup the file as well during `drop()`.
    async fn make_and_open_file(
        block_size: u64,
        generation: Generation,
        encoded_file_path: EncodedFilePath,
    ) -> Result<(Self, FileSlot, OsString), Error> {
        let temp_full_path = encoded_file_path.get_file_path().to_os_string();
        let temp_file_result = fs::create_file(temp_full_path.clone())
            .or_else(|mut err| async {
                let remove_result = fs::remove_file(&temp_full_path).await.err_tip(|| {
                    format!(
                        "Failed to remove file {} in filesystem store",
                        temp_full_path.display()
                    )
                });
                if let Err(remove_err) = remove_result {
                    err = err.merge(remove_err);
                }
                warn!(?err, ?block_size, ?temp_full_path, "Failed to create file",);
                Err(err).err_tip(|| {
                    format!(
                        "Failed to create {} in filesystem store",
                        temp_full_path.display()
                    )
                })
            })
            .await?;

        Ok((
            <Self as FileEntry>::create(
                0, /* Unknown yet, we will fill it in later */
                block_size,
                generation,
                RwLock::new(encoded_file_path),
            ),
            temp_file_result,
            temp_full_path,
        ))
    }

    fn data_size(&self) -> u64 {
        self.data_size
    }

    fn generation(&self) -> Generation {
        self.generation
    }

    fn data_size_mut(&mut self) -> &mut u64 {
        &mut self.data_size
    }

    fn size_on_disk(&self) -> u64 {
        self.data_size.div_ceil(self.block_size) * self.block_size
    }

    fn get_encoded_file_path(&self) -> &RwLock<EncodedFilePath> {
        &self.encoded_file_path
    }

    fn read_file_part(
        &self,
        offset: u64,
        length: u64,
    ) -> impl Future<Output = Result<Take<FileSlot>, Error>> + Send {
        self.get_file_path_locked(move |full_content_path| async move {
            let file = fs::open_file(&full_content_path, offset, length)
                .await
                .err_tip(|| {
                    format!(
                        "Failed to open file in filesystem store {}",
                        full_content_path.display()
                    )
                })?;
            Ok(file)
        })
    }

    async fn get_file_path_locked<
        T,
        Fut: Future<Output = Result<T, Error>> + Send,
        F: FnOnce(OsString) -> Fut + Send,
    >(
        &self,
        handler: F,
    ) -> Result<T, Error> {
        let encoded_file_path = self.get_encoded_file_path().read().await;
        handler(encoded_file_path.get_file_path().to_os_string()).await
    }
}

impl Debug for FileEntryImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("FileEntryImpl")
            .field("data_size", &self.data_size)
            .field("encoded_file_path", &"<behind mutex>")
            .finish()
    }
}

fn make_temp_digest(mut digest: DigestInfo) -> DigestInfo {
    static DELETE_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hash = *digest.packed_hash();
    hash[24..].clone_from_slice(
        &DELETE_FILE_COUNTER
            .fetch_add(1, Ordering::Relaxed)
            .to_le_bytes(),
    );
    digest.set_packed_hash(*hash);
    digest
}

pub fn make_temp_key(key: &StoreKey) -> StoreKey<'static> {
    StoreKey::Digest(make_temp_digest(key.borrow().into_digest()))
}

/// Group-commit flush coalescer (macOS only).
///
/// On macOS, `File::sync_all` issues `fcntl(F_FULLFSYNC)` — a full
/// device-cache flush — once per call. The flush is serialized at the
/// device, costs multiple milliseconds, and dominates uploads of many
/// small blobs (a 4,400-tiny-file action input tree spends ~12s of its
/// ~14.5s materialization in these flushes). Linux does not have this
/// problem because concurrent `fsync` calls coalesce inside the
/// filesystem journal's group commit.
///
/// This applies the same group-commit idea in userspace, exploiting an
/// asymmetry in `F_FULLFSYNC`: writing a file's pages to the storage
/// device is per-file work (plain `fsync(2)`, cheap), while the expensive
/// device-cache drain is device-wide by nature. Each writer first pushes
/// its own data to the device with `fsync(2)`, then joins the current
/// commit round; a single long-lived flusher task per DEVICE issues one
/// `F_FULLFSYNC` (on a dedicated sentinel file) per round, covering every
/// previously-`fsync`ed blob at once. Rounds self-batch exactly like a
/// journal: while one flush is running, later writers accumulate into
/// the next round.
///
/// The coalescer (and its flusher task) is per store instance. Sharing
/// one coalescer across all stores on a device would amortize further —
/// the flush is device-wide — but a process-global flusher task cannot
/// safely outlive the tokio runtime that spawned it (multiple runtimes
/// coexist in one process, e.g. one per test); doing this correctly
/// needs a runtime-agnostic flusher (a dedicated OS thread) and is left
/// as a follow-up.
///
/// Durability is identical to per-file `F_FULLFSYNC`: a writer only
/// proceeds (and only renames its blob into the content directory) after
/// a device-cache flush that started after its own `fsync(2)` completed.
#[cfg(target_os = "macos")]
#[derive(Debug)]
struct FlushCoalescer {
    /// The round currently accepting waiters. Swapped out by the flusher
    /// right before it flushes, so writers whose `fsync(2)` finished
    /// after the flush began land in the next round.
    current_round: parking_lot::Mutex<Arc<FlushRound>>,
    /// Wakes the flusher task. Writers notify after subscribing to their
    /// round, so a wakeup can never be observed before its waiter.
    wake: Arc<tokio::sync::Notify>,
    /// Dedicated file the `F_FULLFSYNC` is issued on. A SIBLING of the
    /// store's content path (like the `.exec` variant directory), so
    /// neither the startup scan nor `move_old_cache`'s legacy root sweep
    /// ever sees it.
    sentinel: Arc<std::fs::File>,
    /// Number of device-wide barriers issued. Kept separately from the
    /// per-file `fsync(2)` count so tests can pin the coalescing invariant.
    full_flush_count: Arc<AtomicU64>,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct FlushRound {
    /// Broadcasts the flush outcome to every writer in the round.
    result_tx: tokio::sync::watch::Sender<Option<Result<(), Error>>>,
}

#[cfg(target_os = "macos")]
impl FlushRound {
    fn new() -> Arc<Self> {
        let (result_tx, _) = tokio::sync::watch::channel(None);
        Arc::new(Self { result_tx })
    }
}

/// Guarantees a [`FlushRound`]'s waiters always receive a result: if the
/// flusher task is cancelled at an await point (runtime shutdown), waiters
/// must get an error rather than pend forever — their own `Arc<FlushRound>`
/// keeps the channel alive, so a dropped-sender wakeup can never happen.
#[cfg(target_os = "macos")]
struct SendOnDrop(Option<Arc<FlushRound>>);

#[cfg(target_os = "macos")]
impl SendOnDrop {
    fn finish(mut self, result: Result<(), Error>) {
        if let Some(round) = self.0.take() {
            // Ignore send errors: every waiter may have been cancelled.
            drop(round.result_tx.send(Some(result)));
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for SendOnDrop {
    fn drop(&mut self) {
        if let Some(round) = self.0.take() {
            drop(round.result_tx.send(Some(Err(make_err!(
                Code::Internal,
                "Flush coalescer round task cancelled before completing"
            )))));
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for FlushCoalescer {
    fn drop(&mut self) {
        // Wake the flusher so it observes the dead Weak and exits instead
        // of parking forever.
        self.wake.notify_one();
    }
}

#[cfg(target_os = "macos")]
impl FlushCoalescer {
    /// Creates the coalescer for a store and spawns its flusher task on
    /// the current runtime (the same runtime the store's uploads run on).
    async fn for_content_path(content_path: &str) -> Result<Arc<Self>, Error> {
        // Sibling of `content_path` (never inside it): the startup scan
        // and `move_old_cache` sweep everything under the content root.
        let sentinel_path = format!("{content_path}.flush_sentinel");
        let sentinel = spawn_blocking!("filesystem_store_flush_sentinel_open", move || {
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open(&sentinel_path)
                .map_err(|e| {
                    make_err!(
                        Code::Internal,
                        "Failed to create flush sentinel {sentinel_path}: {e:?}"
                    )
                })
        })
        .await
        .err_tip(|| "Failed to join flush sentinel open task")??;

        let coalescer = Arc::new(Self {
            current_round: parking_lot::Mutex::new(FlushRound::new()),
            wake: Arc::new(tokio::sync::Notify::new()),
            sentinel: Arc::new(sentinel),
            full_flush_count: Arc::new(AtomicU64::new(0)),
        });

        let weak = Arc::downgrade(&coalescer);
        let wake = coalescer.wake.clone();
        background_spawn!("filesystem_store_flush_coalescer", async move {
            loop {
                wake.notified().await;
                let Some(coalescer) = weak.upgrade() else {
                    return;
                };
                coalescer.flush_one_round().await;
                // Drop the strong ref before parking so the store can be
                // torn down while the flusher is idle.
            }
        });
        Ok(coalescer)
    }

    /// Waits until a device-cache flush that started after this call has
    /// completed. Callers must have already `fsync(2)`ed their own file.
    async fn commit(&self) -> Result<(), Error> {
        let round = self.current_round.lock().clone();
        let mut result_rx = round.result_tx.subscribe();
        // Notify strictly after subscribing: the flusher skips rounds
        // with no receivers, so this ordering makes lost wakeups
        // impossible (a permit is stored even while the flusher is busy).
        self.wake.notify_one();
        let result_ref = result_rx
            .wait_for(Option::is_some)
            .await
            .map_err(|_| make_err!(Code::Internal, "Flush coalescer round dropped"))?;
        result_ref
            .clone()
            .unwrap_or_else(|| Err(make_err!(Code::Internal, "Flush round result missing")))
    }

    async fn flush_one_round(&self) {
        let round = self.current_round.lock().clone();
        if round.result_tx.receiver_count() == 0 {
            // Spurious wakeup (e.g. a permit left over from a round that
            // a previous iteration already served).
            return;
        }
        let send_guard = SendOnDrop(Some(round.clone()));

        // Brief accumulation window, only when this round actually has
        // multiple waiters: lets a burst pack more writers into the round
        // before it closes, trading ~3ms of publish latency (about the
        // cost of one device flush) for fewer device flushes. A solo
        // writer on an idle store skips it and pays only its own flush.
        if round.result_tx.receiver_count() > 1 {
            tokio::time::sleep(Duration::from_millis(3)).await;
        }
        // Close the round: writers arriving from here on cannot assume
        // this flush covers them, so they must get a fresh round.
        {
            let mut current = self.current_round.lock();
            if Arc::ptr_eq(&*current, &round) {
                *current = FlushRound::new();
            }
        }
        let sentinel = self.sentinel.clone();
        let full_flush_count = self.full_flush_count.clone();
        let flush_result = spawn_blocking!("filesystem_store_full_flush", move || {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::FileExt;
            // Keep the sentinel dirty so the flush is never a no-op.
            // Best-effort: a failed write (e.g. ENOSPC on a full CoW
            // volume) must not fail the whole round — the fcntl below is
            // what provides the device-cache drain.
            if let Err(err) = sentinel.write_at(b"f", 0) {
                warn!(?err, "Flush sentinel write failed; issuing flush anyway");
            }
            // Count the actual blocking operation, rather than merely counting
            // rounds scheduled by the async task.
            full_flush_count.fetch_add(1, Ordering::Relaxed);
            loop {
                if unsafe { libc::fcntl(sentinel.as_raw_fd(), libc::F_FULLFSYNC) } != -1 {
                    return Ok(());
                }
                let err = std::io::Error::last_os_error();
                // std's sync_all retries EINTR (cvt_r); match it.
                if err.kind() != std::io::ErrorKind::Interrupted {
                    return Err(Error::from(err).append("F_FULLFSYNC failed in flush coalescer"));
                }
            }
        })
        .await
        .unwrap_or_else(|e| {
            Err(Error::from_std_err(Code::Internal, &e).append("Flush coalescer task failed"))
        });
        send_guard.finish(flush_result);
    }
}

#[cfg(unix)]
async fn prepare_executable_dir(content_path: &str) -> Result<bool, Error> {
    fn is_non_writable_error(err: &std::io::Error) -> bool {
        matches!(
            err.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
        )
    }

    let executable_dir = format!("{content_path}{EXECUTABLE_DIR_SUFFIX}");
    let executable_digest_dir = format!("{executable_dir}/{DIGEST_FOLDER_V2}");
    spawn_blocking!("filesystem_store_prepare_executable_dir", move || {
        match std::fs::remove_dir_all(&executable_dir) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            // A read-only cache must remain available for ordinary reads. Do
            // not reuse the directory, though: a surviving executable variant
            // may be stale or torn after an unclean shutdown.
            Err(err) if is_non_writable_error(&err) => return Ok(false),
            Err(err) => {
                return Err(Error::from(err)
                    .append(format!("Failed to clear executable dir {executable_dir}")));
            }
        }

        match std::fs::create_dir_all(&executable_digest_dir) {
            Ok(()) => Ok(true),
            Err(err) if is_non_writable_error(&err) => Ok(false),
            Err(err) => Err(Error::from(err).append(format!(
                "Failed to create executable dir {executable_digest_dir}"
            ))),
        }
    })
    .await
    .err_tip(|| "Failed to join executable-dir preparation task")?
}

impl LenEntry for FileEntryImpl {
    #[inline]
    fn len(&self) -> u64 {
        self.size_on_disk()
    }

    /// The file's actual length; `len()` is rounded up to whole blocks.
    #[inline]
    fn content_len(&self) -> u64 {
        self.data_size
    }

    fn is_empty(&self) -> bool {
        self.data_size == 0
    }

    // unref() only triggers when an item is removed from the eviction_map. It is possible
    // that another place in code has a reference to `FileEntryImpl` and may later read the
    // file. To support this edge case, we first move the file to a temp file and point
    // target file location to the new temp file. `unref()` should only ever be called once.
    #[inline]
    async fn unref(&self) {
        let mut encoded_file_path = self.encoded_file_path.write().await;
        if encoded_file_path.path_type == PathType::Temp {
            // We are already a temp file that is now marked for deletion on drop.
            // This is very rare, but most likely the rename into the content path failed.
            warn!(
                key = ?encoded_file_path.key,
                "File is already a temp file",
            );
            return;
        }
        let from_path = encoded_file_path.get_file_path();
        let new_key = make_temp_key(&encoded_file_path.key);

        let to_path = to_full_path_from_key(
            &encoded_file_path.shared_context.temp_path,
            &new_key,
            encoded_file_path.generation,
            // Legacy files may survive a failed migration, but only the new
            // temporary directories are guaranteed to exist on this startup.
            Version::V2,
        );

        let renamed = encoded_file_path
            .shared_context
            .retire_blocking({
                let (from_path, to_path) = (from_path.to_os_string(), to_path.clone());
                move || {
                    std::fs::rename(&from_path, &to_path).map_err(|err| {
                        // ENOENT from rename is ambiguous: the source may be
                        // gone, or a directory component of the destination
                        // (the temp dir) may be missing. Confirm the source is
                        // genuinely gone before treating it as benign —
                        // otherwise a removed temp dir would flip an intact
                        // content file to Temp and orphan it on disk.
                        let source_gone = err.kind() == std::io::ErrorKind::NotFound
                            && matches!(
                                std::fs::metadata(&from_path),
                                Err(meta_err) if meta_err.kind() == std::io::ErrorKind::NotFound
                            );
                        (Error::from(err), source_gone)
                    })
                }
            })
            .await
            .unwrap_or_else(|err| Err((err, false)));
        if let Err((err, source_gone)) = renamed {
            if source_gone {
                // The file is already gone — typically another thread's
                // eviction beat us, or the entry never got its file on
                // disk. Benign here, and it dominates log volume under
                // heavy write+evict concurrency, so keep it at `debug`.
                // Mark the entry Temp (as a successful rename would) so a
                // repeat unref is a no-op and drop stops claiming the
                // content path.
                debug!(
                    key = ?encoded_file_path.key,
                    "Failed to rename file (already gone, treating as benign)",
                );
                encoded_file_path.path_type = PathType::Temp;
                encoded_file_path.key = new_key;
                encoded_file_path.version = Version::V2;
            } else {
                // Either a non-ENOENT failure (EACCES, EXDEV, EBUSY, …) or
                // ENOENT with the source still present (missing temp dir).
                // The content file is intact; leave the entry as Content.
                warn!(
                    key = ?encoded_file_path.key,
                    ?from_path,
                    ?to_path,
                    ?err,
                    "Failed to rename file",
                );
            }
        } else {
            debug!(
                key = ?encoded_file_path.key,
                ?from_path,
                ?to_path,
                "Renamed file (unref)",
            );
            encoded_file_path.path_type = PathType::Temp;
            encoded_file_path.key = new_key;
            encoded_file_path.version = Version::V2;
        }
    }
}

#[inline]
fn digest_from_filename(file_name: &str) -> Result<DigestInfo, Error> {
    let (hash, size) = file_name.split_once('-').err_tip(|| "")?;
    let size = size.parse::<i64>()?;
    DigestInfo::try_new(hash, size)
}

fn key_from_file_v1(file_name: &str, file_type: FileType) -> Result<StoreKey<'_>, Error> {
    match file_type {
        FileType::String => Ok(StoreKey::new_str(file_name)),
        FileType::Digest => digest_from_filename(file_name).map(StoreKey::Digest),
    }
}

pub fn key_and_generation_from_file_v2(
    file_name: &str,
    file_type: FileType,
) -> Result<(StoreKey<'_>, Generation), Error> {
    let (key, generation) = file_name.rsplit_once('-').err_tip(|| "")?;
    let generation = Generation::new(generation.parse::<u64>()?);

    let key = match file_type {
        FileType::String => StoreKey::new_str(key),
        FileType::Digest => digest_from_filename(key).map(StoreKey::Digest)?,
    };

    Ok((key, generation))
}

/// The number of files to read the metadata for at the same time when running
/// `add_files_to_cache`.
const SIMULTANEOUS_METADATA_READS: usize = 200;

type FsEvictingMap<'a, Fe> =
    EvictingMap<StoreKeyBorrow, StoreKey<'a>, Arc<Fe>, SystemTime, RemoveCallbackHolder>;

async fn add_files_to_cache<Fe: FileEntry>(
    evicting_map: &FsEvictingMap<'_, Fe>,
    anchor_time: &SystemTime,
    shared_context: &Arc<SharedContext>,
    block_size: u64,
    rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    migrate: bool,
) -> Result<Generation, Error> {
    #[expect(clippy::too_many_arguments)]
    async fn process_entry<Fe: FileEntry>(
        evicting_map: &FsEvictingMap<'_, Fe>,
        file_name: &str,
        file_type: FileType,
        version: Version,
        atime: SystemTime,
        data_size: u64,
        block_size: u64,
        anchor_time: &SystemTime,
        shared_context: &Arc<SharedContext>,
    ) -> Result<Generation, Error> {
        let (key, generation) = match version {
            Version::Flat | Version::V1 => {
                (key_from_file_v1(file_name, file_type)?, Generation::new(0))
            }
            Version::V2 => key_and_generation_from_file_v2(file_name, file_type)?,
        };

        let file_entry = Arc::new(Fe::create(
            data_size,
            block_size,
            generation,
            RwLock::new(EncodedFilePath {
                shared_context: shared_context.clone(),
                path_type: PathType::Content,
                key: key.borrow().into_owned(),
                generation,
                version,
            }),
        ));
        let time_since_anchor = if let Ok(d) = anchor_time.duration_since(atime) {
            d
        } else {
            warn!(
                %file_name,
                atime = %humantime::format_rfc3339(atime),
                anchor_time = %humantime::format_rfc3339(*anchor_time),
                "File access time newer than FilesystemStore start time",
            );
            Duration::ZERO
        };
        let (inserted, _) = evicting_map
            .insert_with_time_if(
                key.into_owned().into(),
                file_entry.clone(),
                |present_entry, new_entry| present_entry.generation() < new_entry.generation(),
                i32::try_from(time_since_anchor.as_secs()).unwrap_or(i32::MAX),
            )
            .await;
        if !inserted {
            // Older generations can survive a crash before deferred cleanup.
            // Retire rejected files too so the next restart cannot revive them.
            file_entry.unref().await;
        }
        Ok(generation)
    }

    async fn read_files(
        folder: Option<&str>,
        shared_context: &SharedContext,
    ) -> Result<Vec<(String, SystemTime, u64, bool)>, Error> {
        // Note: In Dec 2024 this is for backwards compatibility with the old
        // way files were stored on disk. Previously all files were in a single
        // folder regardless of the StoreKey type. This allows old versions of
        // nativelink file version to be upgraded at startup time.
        // This logic can be removed once more time has passed.
        let read_dir = folder.map_or_else(
            || format!("{}/", shared_context.content_path),
            |folder| format!("{}/{folder}/", shared_context.content_path),
        );

        let (_permit, dir_handle) = match fs::read_dir(read_dir).await {
            Ok(dir_handle) => dir_handle.into_inner(),
            Err(err) if err.code == Code::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(err).err_tip(
                    || "Failed opening content directory for iterating in filesystem store",
                );
            }
        };

        let read_dir_stream = ReadDirStream::new(dir_handle);
        read_dir_stream
            .map(|dir_entry| async move {
                let dir_entry = dir_entry.unwrap();
                let file_name = dir_entry.file_name().into_string().unwrap();
                let metadata = dir_entry
                    .metadata()
                    .await
                    .err_tip(|| "Failed to get metadata in filesystem store")?;
                // We need to filter out folders - we do not want to try to cache
                // the per-version key folders.
                let is_file = metadata.is_file()
                    || ![
                        STR_FOLDER_V2,
                        DIGEST_FOLDER_V2,
                        STR_FOLDER_V1,
                        DIGEST_FOLDER_V1,
                    ]
                    .contains(&file_name.as_str());
                // Using access time is not perfect, but better than random. We do not update the
                // atime when a file is actually "touched", we rely on whatever the filesystem does
                // when we read the file (usually update on read).
                let atime = metadata
                    .accessed()
                    .or_else(|_| metadata.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                Result::<(String, SystemTime, u64, bool), Error>::Ok((
                    file_name,
                    atime,
                    metadata.len(),
                    is_file,
                ))
            })
            .buffer_unordered(SIMULTANEOUS_METADATA_READS)
            .try_collect()
            .await
    }

    /// Best effort: a failure leaves the source where it is, so the loader
    /// still picks it up and a later startup retries.
    fn migrate_file(
        from_file: &OsStr,
        to_file: &OsStr,
        rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    ) {
        if let Err(err) = rename_fn(from_file, to_file) {
            warn!(?from_file, ?to_file, ?err, "Failed to migrate file");
        } else {
            debug!(?from_file, ?to_file, "Migrated file");
        }
    }

    /// Note: In Dec 2024 this is for backwards compatibility with the old
    /// way files were stored on disk. Previously all files were in a single
    /// folder regardless of the [`StoreKey`] type.
    async fn migrate_old_cache_1(
        shared_context: &Arc<SharedContext>,
        rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    ) -> Result<(), Error> {
        let file_infos = read_files(None, shared_context).await?;
        let from_path = &shared_context.content_path;
        let to_path = format!("{}/{DIGEST_FOLDER_V2}", shared_context.content_path);

        for (file_name, _, _, _) in file_infos.into_iter().filter(|x| x.3) {
            migrate_file(
                &OsString::from(format!("{from_path}/{file_name}")),
                &OsString::from(format!("{to_path}/{file_name}-0")),
                rename_fn,
            );
        }
        Ok(())
    }

    /// Moves [`Version::V1`] folders into [`Version::V2`].
    async fn migrate_old_cache_2(
        shared_context: &Arc<SharedContext>,
        rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    ) -> Result<(), Error> {
        for (legacy_folder, folder) in [
            (DIGEST_FOLDER_V1, DIGEST_FOLDER_V2),
            (STR_FOLDER_V1, STR_FOLDER_V2),
        ] {
            let file_infos = read_files(Some(legacy_folder), shared_context).await?;
            let from_path = format!("{}/{legacy_folder}", shared_context.content_path);
            let to_path = format!("{}/{folder}", shared_context.content_path);

            for (file_name, _, _, _) in file_infos.into_iter().filter(|x| x.3) {
                migrate_file(
                    &OsString::from(format!("{from_path}/{file_name}")),
                    &OsString::from(format!("{to_path}/{file_name}-0")),
                    rename_fn,
                );
            }
        }
        Ok(())
    }

    async fn add_folder_to_cache<Fe: FileEntry>(
        evicting_map: &FsEvictingMap<'_, Fe>,
        anchor_time: &SystemTime,
        shared_context: &Arc<SharedContext>,
        block_size: u64,
        folder: &str,
        file_type: FileType,
        version: Version,
    ) -> Result<Generation, Error> {
        let mut file_infos = read_files(Some(folder), shared_context).await?;
        // Load older generations first. Otherwise cache pressure could evict a
        // newer generation before an older file for that key is encountered.
        if version == Version::V2 {
            file_infos.sort_by_cached_key(|(file_name, ..)| {
                key_and_generation_from_file_v2(file_name, file_type)
                    .map_or(Generation::new(0), |(_, generation)| generation)
            });
        }

        let path_root = format!("{}/{folder}", shared_context.content_path);

        let mut max_generation = 0;

        for (file_name, atime, data_size, _) in file_infos.into_iter().filter(|x| x.3) {
            let result = process_entry(
                evicting_map,
                &file_name,
                file_type,
                version,
                atime,
                data_size,
                block_size,
                anchor_time,
                shared_context,
            )
            .await;

            match result {
                Ok(generation) => {
                    max_generation = cmp::max(max_generation, generation.inner());
                }
                Err(err) => {
                    warn!(?file_name, ?err, "Failed to add file to eviction cache",);
                    // Ignore result.
                    drop(fs::remove_file(format!("{path_root}/{file_name}")).await);
                }
            }
        }
        Ok(Generation::new(max_generation))
    }

    if migrate {
        migrate_old_cache_1(shared_context, rename_fn).await?;
        migrate_old_cache_2(shared_context, rename_fn).await?;
    }

    let mut max_generation = 0;
    for (folder, file_type, version) in [
        ("", FileType::Digest, Version::Flat),
        (DIGEST_FOLDER_V1, FileType::Digest, Version::V1),
        (STR_FOLDER_V1, FileType::String, Version::V1),
        (DIGEST_FOLDER_V2, FileType::Digest, Version::V2),
        (STR_FOLDER_V2, FileType::String, Version::V2),
    ] {
        let generation = add_folder_to_cache(
            evicting_map,
            anchor_time,
            shared_context,
            block_size,
            folder,
            file_type,
            version,
        )
        .await?;
        max_generation = cmp::max(max_generation, generation.inner());
    }

    Ok(Generation::new(max_generation.checked_add(1).ok_or_else(
        || {
            make_err!(
                Code::ResourceExhausted,
                "Filesystem generation counter exhausted"
            )
        },
    )?))
}

async fn prune_temp_path(temp_path: &str) -> Result<(), Error> {
    async fn prune_temp_inner(temp_path: &str, subpath: &str) -> Result<(), Error> {
        let (_permit, dir_handle) = match fs::read_dir(format!("{temp_path}/{subpath}")).await {
            Ok(dir_handle) => dir_handle.into_inner(),
            Err(err) if err.code == Code::NotFound => return Ok(()),
            Err(err) => {
                return Err(err).err_tip(|| {
                    "Failed opening temp directory to prune partial downloads in filesystem store"
                });
            }
        };

        let mut read_dir_stream = ReadDirStream::new(dir_handle);
        while let Some(dir_entry) = read_dir_stream.next().await {
            let path = dir_entry?.path();
            if let Err(err) = fs::remove_file(&path).await {
                warn!(?path, ?err, "Failed to delete file",);
            }
        }
        Ok(())
    }

    for folder in [
        STR_FOLDER_V2,
        DIGEST_FOLDER_V2,
        STR_FOLDER_V1,
        DIGEST_FOLDER_V1,
    ] {
        prune_temp_inner(temp_path, folder).await?;
    }
    Ok(())
}

// Sometimes we get files to emplace that are identical to the existing files
// Due to the evict/remove/replace cycle taking some amount of time, we actually
// want to drop these
// Return value is "is duplicate"
pub async fn check_duplicate_files<Fe>(
    evicting_map: &Arc<FsEvictingMap<'_, Fe>>,
    key: &StoreKey<'static>,
    entry: &Arc<Fe>,
) -> Result<bool, Error>
where
    Fe: FileEntry,
{
    check_duplicate_files_with(evicting_map, key, entry, None).await
}

/// [`check_duplicate_files`], reading the new file through `open_temp_file`
/// when its writer still holds it open. The check then needs one more
/// open-file permit, for the existing file, rather than two.
async fn check_duplicate_files_with<Fe>(
    evicting_map: &Arc<FsEvictingMap<'_, Fe>>,
    key: &StoreKey<'static>,
    entry: &Arc<Fe>,
    open_temp_file: Option<&mut FileSlot>,
) -> Result<bool, Error>
where
    Fe: FileEntry,
{
    let temp_file_encoded_file_path = entry.get_encoded_file_path().write().await;
    let maybe_existing_item = evicting_map.get(&key.borrow().into_owned()).await;
    if let Some(existing_item) = maybe_existing_item {
        if Arc::ptr_eq(entry, &existing_item) {
            warn!("Tried to check duplicate of an entry we already have!");
            return Ok(true);
        }
        let existing_item_encoded_file_path = existing_item.get_encoded_file_path().write().await;
        if entry.data_size() == existing_item.data_size() {
            const CHUNK_SIZE: usize = 16 * 1024; // 16kb chunks, kinda picked out of the air
            let file_length = entry.data_size();
            let existing_path = existing_item_encoded_file_path.get_file_path();
            let temp_path = temp_file_encoded_file_path.get_file_path();
            trace!(?existing_path, ?temp_path, "Checking duplicate files");
            let mut opened_temp_file = None;
            let temp_file: &mut (dyn AsyncRead + Unpin + Send) = match open_temp_file {
                Some(file) => {
                    if let Err(err) = file.seek(SeekFrom::Start(0)).await {
                        warn!(
                            ?err,
                            ?temp_path,
                            "Failed to rewind temp file, skipping duplicate check"
                        );
                        return Ok(false);
                    }
                    file
                }
                None => opened_temp_file.insert(fs::open_file(&temp_path, 0, file_length).await?),
            };
            let mut existing_file = match fs::open_file(&existing_path, 0, file_length).await {
                Ok(file) => file,
                Err(err) if err.code == Code::NotFound => {
                    warn!(
                        ?key,
                        ?err,
                        "Map/disk divergence in check_duplicate_files: dropping stale entry",
                    );
                    // Removal calls unref(), which needs the path lock. Release
                    // both locks and the open file before running callbacks.
                    drop(opened_temp_file);
                    drop(existing_item_encoded_file_path);
                    drop(temp_file_encoded_file_path);
                    evicting_map
                        .remove_if(key, |map_entry| Arc::ptr_eq(map_entry, &existing_item))
                        .await;
                    return Ok(false);
                }
                Err(err) => return Err(err),
            };

            let mut temp_buffer: [u8; CHUNK_SIZE] = [0; CHUNK_SIZE];
            let mut existing_buffer: [u8; CHUNK_SIZE] = [0; CHUNK_SIZE];
            for offset in (0..file_length).step_by(CHUNK_SIZE) {
                let buffer_size = if offset + (CHUNK_SIZE as u64) <= file_length {
                    CHUNK_SIZE
                } else if file_length < CHUNK_SIZE as u64 {
                    usize::try_from(file_length)
                        .expect("Always succeeds because file_length < 16384")
                } else {
                    usize::try_from(file_length - offset).expect("Always succeeds because offset < file_length, and offset-file_length must be < 16384")
                };
                if let Err(err) = temp_file.read_exact(&mut temp_buffer[0..buffer_size]).await {
                    warn!(
                        ?err,
                        ?temp_path,
                        file_length,
                        offset,
                        buffer_size,
                        "Failed to read temp file, skipping duplicate check"
                    );
                    return Ok(false);
                }
                if let Err(err) = existing_file
                    .read_exact(&mut existing_buffer[0..buffer_size])
                    .await
                {
                    warn!(
                        ?err,
                        ?existing_path,
                        file_length,
                        offset,
                        buffer_size,
                        "Failed to read existing, skipping duplicate check"
                    );
                    return Ok(false);
                }
                if temp_buffer.ne(&existing_buffer) {
                    trace!(
                        ?existing_path,
                        ?temp_path,
                        "Files are different, so non-duplicate"
                    );
                    return Ok(false);
                }
            }
            trace!(
                ?existing_path,
                ?temp_path,
                "Identical files, so don't need to edit, skipping emplace"
            );
            drop(existing_file);
            drop(opened_temp_file);
            drop(existing_item_encoded_file_path);
            drop(temp_file_encoded_file_path);
            // Comparison can overlap eviction. An identical retired file no
            // longer satisfies this upload unless it is still the mapped entry.
            return Ok(evicting_map
                .get(key)
                .await
                .is_some_and(|current| Arc::ptr_eq(&current, &existing_item)));
        }
        trace!(
            entry_data_size = entry.data_size(),
            existing_data_size = existing_item.data_size(),
            existing_path = ?existing_item_encoded_file_path.get_file_path(),
            "Different data sizes, so non-duplicate"
        );
    } else {
        trace!(
            temp_file = ?temp_file_encoded_file_path.get_file_path(),
            "No existing entry, so not duplicate"
        );
    }
    Ok(false)
}

/// What is left once a file is published, none of it part of publishing it.
#[must_use = "displaced or superseded files are only retired when this is run"]
#[derive(Debug)]
enum AfterPublication<Fe> {
    /// The file is in the map. Unref the entries its insertion replaced or
    /// evicted.
    Inserted(InsertCleanup<Arc<Fe>>),
    /// A newer generation of the key was published first, so the map kept
    /// that one. Retire this file too, so it cannot leak or reappear after a
    /// restart.
    Superseded(Arc<Fe>),
}

impl<Fe: FileEntry> AfterPublication<Fe> {
    /// Retires the displaced or superseded files. This waits for the store's
    /// retirement admission (`unref()` renames them into the temp directory;
    /// see [`SharedContext::retire_blocking`]), never for an open-file permit.
    async fn run(self) {
        match self {
            Self::Inserted(cleanup) => drop(cleanup.run().await),
            Self::Superseded(entry) => entry.unref().await,
        }
    }
}

/// Renames `entry`'s temp file to its content path for `key`, then inserts
/// `entry` into `evicting_map` if no newer generation of `key` is there.
/// Publishing waits for no open-file permit: `rename_fn` runs directly on the
/// blocking pool, and retiring the files the insertion displaced is left to
/// the returned [`AfterPublication`].
async fn publish_file<Fe: FileEntry>(
    evicting_map: &FsEvictingMap<'static, Fe>,
    rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    key: StoreKey<'static>,
    entry: Arc<Fe>,
) -> Result<AfterPublication<Fe>, Error> {
    // Each generation owns a distinct content path. Publish the file before
    // inserting it into the map, so readers only observe completed renames.
    // A delayed unref of another generation cannot touch this file.
    let mut encoded_file_path = entry.get_encoded_file_path().write().await;

    let final_path = get_file_path_raw(
        &PathType::Content,
        encoded_file_path.shared_context.as_ref(),
        &key,
        encoded_file_path.generation,
        encoded_file_path.version,
    );

    let from_path = encoded_file_path.get_file_path();

    // Renames between two directories on one filesystem are serialized in
    // the kernel and can block for a long time, so the rename runs on the
    // blocking pool rather than on a runtime thread, which would stall every
    // other task scheduled on it.
    let result = spawn_blocking!("filesystem_store_emplace_rename", {
        let (from_path, final_path) = (
            from_path.clone().into_owned(),
            final_path.clone().into_owned(),
        );
        move || (rename_fn)(&from_path, &final_path)
    })
    .await
    .map_err(Error::from)
    .and_then(|renamed| renamed.map_err(Error::from))
    .err_tip(|| {
        format!(
            "Failed to rename temp file to final path {}",
            final_path.display()
        )
    });

    if let Err(err) = result {
        error!(?err, ?from_path, ?final_path, "Failed to rename file",);
        // This entry was never visible in the map. Drop cleans up its
        // temporary file and the previous generation remains usable.
        return Err(err);
    }
    trace!(?key, "Finished emplace file");
    encoded_file_path.path_type = PathType::Content;
    encoded_file_path.key = key.clone();
    // Insertion can unref this entry immediately under cache pressure.
    drop(encoded_file_path);
    let (inserted, cleanup) = evicting_map.insert_if_deferring_cleanup(
        key.clone().into(),
        entry.clone(),
        |present, new| present.generation() < new.generation(),
    );
    if inserted {
        Ok(AfterPublication::Inserted(cleanup))
    } else {
        info!(%key, "Newer generation already emplaced, dropping");
        drop(cleanup);
        Ok(AfterPublication::Superseded(entry))
    }
}

/// Deletes a digest's `.exec` variant (see
/// [`FilesystemStore::get_executable_hardlink_source`]) when that digest is
/// evicted or replaced in the primary CAS `evicting_map`. Without this, the
/// `.exec` directory is invisible to `max_bytes` and is only ever cleared by
/// the startup `remove_dir_all`, so it grows without bound at runtime (#2474).
/// Tying its lifetime to the primary entry instead bounds total disk use to
/// roughly `2 * max_bytes` in the worst case (every blob also executable).
///
/// A variant is named by its digest alone and serves whichever generation of
/// the digest is resident, while a callback names only the key it was
/// registered for. A callback can run long after its entry left the map (it
/// waits for retirement admission behind earlier retirements), by which time
/// the digest may be resident again with a variant a worker is about to
/// hardlink. So the callback deletes the variant only when no generation of
/// the digest is resident, decided under the digest's
/// [`ExecutableVariantLease`] and held through the unlink:
///
/// - Materialization holds the same lease, so the unlink never races a
///   variant being published.
/// - Fast-path readers return an existing variant only for a digest they saw
///   resident and then saw with its lease free
///   ([`FilesystemStore::variants_servable_without_lease`]). Such a reader
///   looked before this callback took the lease, and a caller that keeps the
///   digest resident (a worker leases its inputs) is still resident when the
///   callback checks, so the variant is kept. A lease on a digest that is not
///   resident does not protect its variant: this callback may already be
///   waiting to delete it, so such a reader takes the per-digest path, whose
///   caller populates the digest first.
/// - A resident generation's own removal later retires the variant.
///
/// Cancelled before its unlink starts, the callback deletes nothing: the
/// variant stays until the digest's next removal or the startup wipe, and a
/// newer generation never loses its variant to it.
#[cfg(unix)]
#[derive(Debug)]
struct ExecutableVariantRemover<Fe: FileEntry> {
    content_path: String,
    shared_context: Arc<SharedContext>,
    executable_locks: Arc<ExecutableLocks>,
    /// Weak because the map owns this callback.
    evicting_map: Weak<FsEvictingMap<'static, Fe>>,
}

/// What an [`ExecutableVariantRemover`] did with a variant.
#[cfg(unix)]
#[derive(Debug)]
enum VariantRetirement {
    Deleted,
    /// No variant was ever materialized, or it was already deleted.
    Absent,
    /// A generation of the digest is resident again and owns the variant.
    KeptForResidentGeneration,
}

#[cfg(unix)]
impl<Fe: FileEntry> RemoveItemCallback for ExecutableVariantRemover<Fe> {
    fn callback<'a>(
        &'a self,
        store_key: StoreKey<'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            let StoreKey::Digest(digest) = store_key else {
                return;
            };
            let variant_path = format!(
                "{}{EXECUTABLE_DIR_SUFFIX}/{DIGEST_FOLDER_V2}/{digest}",
                self.content_path
            );
            let lease = ExecutableVariantLease::acquire(&self.executable_locks, digest).await;
            let evicting_map = self.evicting_map.clone();
            let retired = self
                .shared_context
                .retire_blocking({
                    let variant_path = variant_path.clone();
                    move || {
                        // Released only once the unlink has really ended, even
                        // if this callback is cancelled meanwhile.
                        let _lease = lease;
                        let key = StoreKey::Digest(digest);
                        if evicting_map
                            .upgrade()
                            .is_some_and(|map| map.is_resident(&key))
                        {
                            return Ok(VariantRetirement::KeptForResidentGeneration);
                        }
                        match std::fs::remove_file(&variant_path) {
                            Ok(()) => Ok(VariantRetirement::Deleted),
                            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                                Ok(VariantRetirement::Absent)
                            }
                            Err(err) => Err(Error::from(err)),
                        }
                    }
                })
                .await
                .and_then(|retired| retired);
            match retired {
                Ok(VariantRetirement::Deleted) => debug!(
                    ?variant_path,
                    "Deleted executable variant for evicted digest"
                ),
                Ok(VariantRetirement::Absent) => {}
                Ok(VariantRetirement::KeptForResidentGeneration) => debug!(
                    ?variant_path,
                    "Kept executable variant of a digest that is resident again"
                ),
                Err(err) => warn!(
                    ?variant_path,
                    ?err,
                    "Failed to delete executable variant for evicted digest"
                ),
            }
        })
    }
}

#[derive(Debug, MetricsComponent)]
pub struct FilesystemStore<Fe: FileEntry = FileEntryImpl> {
    #[metric]
    shared_context: Arc<SharedContext>,
    #[metric(group = "evicting_map")]
    evicting_map: Arc<FsEvictingMap<'static, Fe>>,
    #[metric(help = "Block size of the configured filesystem")]
    block_size: u64,
    #[metric(help = "Size of the configured read buffer size")]
    read_buffer_size: usize,
    /// See [`FilesystemSpec::evict_page_cache`]. When false (the default) the
    /// per-blob `posix_fadvise(DONTNEED)` calls are skipped.
    evict_page_cache: bool,
    weak_self: Weak<Self>,
    rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    /// Limits concurrent write operations to prevent disk I/O saturation.
    /// A permit covers a write's flush, fsync and rename, so it is owned by
    /// the task doing that work (see [`FilesystemStore::finalize_temp_file`]).
    write_semaphore: Option<Arc<Semaphore>>,
    /// A monotonic counter by which we stamp newly added files.
    next_generation: AtomicU64,
    /// See [`FlushCoalescer`]: amortizes the per-blob `F_FULLFSYNC` cost
    /// across concurrent uploads without weakening durability. `None` when
    /// the sentinel could not be created (e.g. read-only content volume);
    /// uploads then fall back to per-file `sync_all`, matching the old
    /// behavior (and such stores fail at write time anyway).
    #[cfg(target_os = "macos")]
    flush_coalescer: Option<Arc<FlushCoalescer>>,
    /// Per-digest single-flight locks guarding creation of the executable
    /// variant in `{content_path}.exec`, so each variant's writable fd is
    /// opened exactly once. See [`ExecutableVariantLease`].
    #[cfg(unix)]
    executable_locks: Arc<ExecutableLocks>,
    /// False when the startup wipe could not safely remove old executable
    /// variants from a read-only filesystem. Ordinary CAS reads remain
    /// available, but executable hardlink sources must not reuse stale files.
    #[cfg(unix)]
    executable_variants_enabled: bool,
}

/// The per-digest executable-variant locks. The outer lock is sync and only
/// ever held to get, insert, check or remove a per-digest entry — never across
/// I/O.
#[cfg(unix)]
type ExecutableLocks = std::sync::Mutex<HashMap<DigestInfo, Arc<Mutex<()>>>>;

/// Exclusive right to materialize or delete one digest's executable variant.
/// Materialization and [`ExecutableVariantRemover`] both hold it, and readers
/// that skip it check that no one holds it ([`FilesystemStore::variants_servable_without_lease`]).
///
/// A lease holds the digest's lock *and* that lock is still the digest's
/// entry in [`ExecutableLocks`]. Only a lease removes the entry, when it is
/// dropped and before it releases the lock, so the map stays bounded. A
/// caller that waited on a lock whose lease has since retired it finds the
/// entry gone or replaced and queues on the current lock instead. Holding a
/// retired lock therefore never licenses a write: at most one writer per
/// digest, including between an old waiter and a newcomer.
///
/// Every entry therefore belongs to a lease: whoever inserts an entry locks
/// its fresh lock in the same poll, or else queues behind the caller that took
/// it first, whose lease then retires the entry. Cancelling a queued caller
/// only drops its claim on the lock.
///
/// The lease moves into the blocking task that does the copy or the unlink.
/// That task runs to completion even when its caller is cancelled, and the
/// lease is released only when the task — the actual writer — has finished.
#[cfg(unix)]
struct ExecutableVariantLease {
    locks: Arc<ExecutableLocks>,
    digest: DigestInfo,
    guard: MutexGuardArc<()>,
}

#[cfg(unix)]
impl ExecutableVariantLease {
    async fn acquire(locks: &Arc<ExecutableLocks>, digest: DigestInfo) -> Self {
        loop {
            let guard = locks
                .lock()
                .expect("executable_locks poisoned")
                .entry(digest)
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .lock_arc();
            let guard = guard.await;
            let current = locks
                .lock()
                .expect("executable_locks poisoned")
                .get(&digest)
                .is_some_and(|entry| Arc::ptr_eq(entry, MutexGuardArc::source(&guard)));
            if current {
                return Self {
                    locks: locks.clone(),
                    digest,
                    guard,
                };
            }
            // The lease we waited behind retired this lock; dropping `guard`
            // lets its other waiters notice the same, and we queue on the
            // digest's current lock.
        }
    }
}

#[cfg(unix)]
impl Debug for ExecutableVariantLease {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutableVariantLease")
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
impl Drop for ExecutableVariantLease {
    fn drop(&mut self) {
        let mut locks = self.locks.lock().expect("executable_locks poisoned");
        if locks
            .get(&self.digest)
            .is_some_and(|entry| Arc::ptr_eq(entry, MutexGuardArc::source(&self.guard)))
        {
            locks.remove(&self.digest);
        }
        // `guard` is released after this, once the entry is gone.
    }
}

/// The temporary file of one executable-variant materialization attempt.
///
/// Each attempt creates its own uniquely named file with `O_EXCL`, so no
/// attempt ever opens a file another attempt created — in particular not one
/// a failed attempt left behind read-only. Names are unique within the
/// process; files from an earlier process are gone because executable
/// variants are enabled only after the startup wipe clears the directory.
/// Dropping it removes the file unless it was published by the rename.
#[cfg(unix)]
#[derive(Debug)]
struct ExecutableVariantTemp {
    path: OsString,
    published: bool,
}

#[cfg(unix)]
impl ExecutableVariantTemp {
    fn create(variant_path: &OsStr) -> Result<(Self, std::fs::File), std::io::Error> {
        use std::os::unix::fs::OpenOptionsExt;
        static NEXT_ATTEMPT: AtomicU64 = AtomicU64::new(0);
        let mut path = variant_path.to_os_string();
        path.push(format!(
            ".{}.tmp",
            NEXT_ATTEMPT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)?;
        Ok((
            Self {
                path,
                published: false,
            },
            file,
        ))
    }
}

#[cfg(unix)]
impl Drop for ExecutableVariantTemp {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        if let Err(err) = std::fs::remove_file(&self.path) {
            warn!(
                path = ?self.path,
                ?err,
                "Failed to remove an unpublished executable-variant temporary file"
            );
        }
    }
}

impl<Fe: FileEntry> FilesystemStore<Fe> {
    pub async fn new(spec: &FilesystemSpec) -> Result<Arc<Self>, Error> {
        Self::new_with_timeout_and_rename_fn(spec, |from, to| std::fs::rename(from, to)).await
    }

    pub async fn new_with_timeout_and_rename_fn(
        spec: &FilesystemSpec,
        rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    ) -> Result<Arc<Self>, Error> {
        async fn create_subdirs(path: &str) -> Result<bool, Error> {
            let mut writable = true;
            for folder in [STR_FOLDER_V2, DIGEST_FOLDER_V2] {
                let dir = format!("{path}/{folder}");
                writable &= fs::call_with_permit(move |_| match std::fs::create_dir_all(&dir) {
                    Ok(()) => Ok(true),
                    Err(err) if matches!(err.kind(), std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem) => {
                        warn!(%dir, ?err, "Cannot create directory on read-only volume; retaining legacy files");
                        Ok(false)
                    }
                    Err(err) => Err(Error::from(err).append(format!("Failed to create directory {dir}"))),
                }).await?;
            }
            Ok(writable)
        }

        let now = SystemTime::now();

        let empty_policy = nativelink_config::stores::EvictionPolicy::default();
        let eviction_policy = spec.eviction_policy.as_ref().unwrap_or(&empty_policy);
        let evicting_map = Arc::new(EvictingMap::new(eviction_policy, now));

        // Create the generation-aware temp and content directories.

        let temp_dirs_writable = create_subdirs(&spec.temp_path).await?;
        let content_dirs_writable = create_subdirs(&spec.content_path).await?;
        // Nothing to migrate into; skip the scan rather than warn per file.
        let migrate = temp_dirs_writable && content_dirs_writable;

        // Executable-variant directory: a sibling of `content_path` holding
        // per-digest 0o555 copies used as hardlink sources for executable
        // inputs (see `get_executable_hardlink_source`). Cleared on writable
        // startup — the variants are regenerable and we never want a stale one
        // to leak across runs. If a read-only filesystem prevents the wipe,
        // ordinary CAS reads remain enabled but executable variants do not.
        // Unix-only: the executable bit (and the ETXTBSY race it guards
        // against) does not apply on Windows.
        let shared_context = Arc::new(SharedContext::new(
            spec.temp_path.clone(),
            spec.content_path.clone(),
        ));
        #[cfg(unix)]
        let executable_locks = Arc::<ExecutableLocks>::default();

        #[cfg(unix)]
        let executable_variants_enabled = {
            let enabled = prepare_executable_dir(&spec.content_path).await?;
            if !enabled {
                warn!(
                    executable_dir = %format!("{}{EXECUTABLE_DIR_SUFFIX}", spec.content_path),
                    "Executable directory is not writable; serving CAS reads with executable variants disabled"
                );
            }
            if enabled {
                // Only register cleanup when executable variants are enabled.
                // Otherwise surviving variants are deliberately quarantined,
                // and repeated deletion warnings would obscure the fallback.
                evicting_map.add_remove_callback(RemoveCallbackHolder::new(Arc::new(
                    ExecutableVariantRemover {
                        content_path: spec.content_path.clone(),
                        shared_context: shared_context.clone(),
                        executable_locks: executable_locks.clone(),
                        evicting_map: Arc::downgrade(&evicting_map),
                    },
                )));
            }
            enabled
        };

        let block_size = if spec.block_size == 0 {
            DEFAULT_BLOCK_SIZE
        } else {
            spec.block_size
        };
        let next_generation = add_files_to_cache(
            evicting_map.as_ref(),
            &now,
            &shared_context,
            block_size,
            rename_fn,
            migrate,
        )
        .await?;
        prune_temp_path(&shared_context.temp_path).await?;

        let read_buffer_size = if spec.read_buffer_size == 0 {
            DEFAULT_BUFF_SIZE
        } else {
            spec.read_buffer_size as usize
        };
        let write_semaphore = if spec.max_concurrent_writes > 0 {
            Some(Arc::new(Semaphore::new(spec.max_concurrent_writes)))
        } else {
            None
        };
        // Never make construction fail over the durability optimization:
        // a read-only content volume must still serve reads, exactly as
        // it did before the coalescer existed.
        #[cfg(target_os = "macos")]
        let flush_coalescer = FlushCoalescer::for_content_path(&shared_context.content_path)
            .await
            .inspect_err(|err| {
                warn!(
                    ?err,
                    "Failed to create flush coalescer; falling back to per-file sync_all"
                );
            })
            .ok();

        Ok(Arc::new_cyclic(|weak_self| Self {
            shared_context,
            evicting_map,
            block_size,
            read_buffer_size,
            evict_page_cache: spec.evict_page_cache,
            weak_self: weak_self.clone(),
            rename_fn,
            next_generation: AtomicU64::new(next_generation.inner()),
            write_semaphore,
            #[cfg(target_os = "macos")]
            flush_coalescer,
            #[cfg(unix)]
            executable_locks,
            #[cfg(unix)]
            executable_variants_enabled,
        }))
    }

    pub fn get_arc(&self) -> Option<Arc<Self>> {
        self.weak_self.upgrade()
    }

    /// Reserve a digest before it is populated so active action inputs cannot
    /// be evicted while they are being materialized. The lease does not make
    /// an absent digest resident: until it is populated, its executable
    /// variant is not served (see [`Self::get_executable_hardlink_source`]).
    pub fn lease_digest(&self, digest: &DigestInfo) {
        let key: StoreKey<'static> = (*digest).into();
        self.evicting_map.lease_key(StoreKeyBorrow::from(key));
    }

    /// Release a batch of action-input leases and trim retained entries once.
    pub async fn release_digests(&self, digests: &[DigestInfo]) {
        self.evicting_map
            .release_keys(digests.iter().map(|digest| StoreKey::Digest(*digest)))
            .await;
    }

    fn get_and_update_generation(&self) -> Result<Generation, Error> {
        self.next_generation
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .map(Generation::new)
            .map_err(|_| {
                make_err!(
                    Code::ResourceExhausted,
                    "Filesystem generation counter exhausted"
                )
            })
    }

    /// Path of the read-only executable (0o555) variant for `digest`.
    #[cfg(unix)]
    fn executable_variant_path(&self, digest: &DigestInfo) -> OsString {
        format!(
            "{}{EXECUTABLE_DIR_SUFFIX}/{DIGEST_FOLDER_V2}/{digest}",
            self.shared_context.content_path
        )
        .into()
    }

    /// Resolves the executable variant for many digests at once, batching the
    /// warm path's existence checks into a single dispatch.
    ///
    /// Every input file carrying the executable bit resolves through here, and
    /// in a toolchain-heavy input tree that is nearly all of them. The `stat(2)`
    /// this needs costs single-digit microseconds on tmpfs, while the permit
    /// acquisition and thread hop to dispatch one cost tens: asking per path
    /// spent ~4.6s per action on a 15k-file tree, and asking once for the tree
    /// spends ~37ms.
    ///
    /// A hit also counts as a use of the blob. The variant lives outside the
    /// eviction index, so a toolchain binary that every action hardlinks from
    /// its variant would otherwise never move in the LRU and be evicted as the
    /// coldest entry of a full store, taking its variant with it under the
    /// next action's batch. One batched lookup refreshes every hit's position.
    /// A variant whose blob is no longer indexed is reported `NotFound`, not
    /// handed out: the caller then fills the fast store again, which puts the
    /// blob back in the index so its next eviction deletes the variant too.
    ///
    /// Each result is otherwise what [`Self::get_executable_hardlink_source`]
    /// returns for its digest, under the same lifetime precondition. That
    /// index check comes after the stat, so on its own it cannot exclude a
    /// variant whose deletion had started before the digest was indexed
    /// again; `materialized_variants` already counts only digests seen
    /// resident before their lease was seen free.
    pub async fn get_executable_hardlink_sources(
        &self,
        digests: &[DigestInfo],
    ) -> Vec<Result<OsString, Error>> {
        let hits = self.materialized_variants(digests).await;
        let hit_keys: Vec<StoreKey<'static>> = digests
            .iter()
            .zip(&hits)
            .filter(|(_, hit)| hit.is_some())
            .map(|(digest, _)| StoreKey::Digest(*digest))
            .collect();
        let mut indexed = vec![None; hit_keys.len()];
        if !hit_keys.is_empty() {
            self.evicting_map
                .sizes_for_keys(hit_keys.iter(), &mut indexed, false /* peek */)
                .await;
        }
        let mut indexed = indexed.into_iter();

        let mut tasks = Vec::with_capacity(digests.len());
        for (digest, hit) in digests.iter().copied().zip(hits) {
            let orphaned = hit.is_some() && indexed.next().flatten().is_none();
            tasks.push(async move {
                match hit {
                    Some(_) if orphaned => Err(make_err!(
                        Code::NotFound,
                        "{digest} has an executable variant on disk but no indexed blob; fill the fast store again"
                    )),
                    Some(path) => Ok(path),
                    None => self.get_executable_hardlink_source(&digest).await,
                }
            });
        }
        futures::stream::iter(tasks)
            .buffered(MAX_CONCURRENT_VARIANT_LOOKUPS)
            .collect()
            .await
    }

    /// Whether a reader that finds each digest's variant may return it without
    /// taking its [`ExecutableVariantLease`]: a generation of the digest is
    /// resident, and then no one holds the lease, that is, no one may be
    /// materializing or deleting the variant. Both must be seen in that order
    /// and *before* looking at the file: see [`ExecutableVariantRemover`],
    /// which keeps the variant only if a generation is resident when it checks
    /// under the lease. Seeing the lease free first would miss a removal that
    /// took the lease, found no generation and is deleting the variant while
    /// the digest is populated again.
    #[cfg(unix)]
    fn variants_servable_without_lease(&self, digests: &[DigestInfo]) -> Vec<bool> {
        let resident: Vec<bool> = digests
            .iter()
            .map(|digest| self.evicting_map.is_resident(&StoreKey::Digest(*digest)))
            .collect();
        let locks = self
            .executable_locks
            .lock()
            .expect("executable_locks poisoned");
        digests
            .iter()
            .zip(resident)
            .map(|(digest, resident)| resident && !locks.contains_key(digest))
            .collect()
    }

    /// Reports which digests already have an executable variant on disk, in one
    /// batched existence check. A digest that is not resident or whose lease is
    /// held counts as a miss, so it takes the per-digest path, which waits for
    /// the lease and needs the blob resident.
    #[cfg(unix)]
    async fn materialized_variants(&self, digests: &[DigestInfo]) -> Vec<Option<OsString>> {
        let paths: Vec<OsString> = digests
            .iter()
            .map(|digest| self.executable_variant_path(digest))
            .collect();
        let servable = self.variants_servable_without_lease(digests);

        // Variants disabled and a failed batch are the same situation: nothing
        // is known to exist, so every digest takes the per-digest path, which
        // reports real errors per file.
        let exists = if self.executable_variants_enabled {
            fs::exists_many(paths.iter().map(Into::into).collect())
                .await
                .ok()
        } else {
            None
        }
        .unwrap_or_else(|| vec![false; paths.len()]);

        paths
            .into_iter()
            .zip(exists)
            .zip(servable)
            .map(|((path, hit), servable)| (hit && servable).then_some(path))
            .collect()
    }

    /// Non-unix never builds executable variants, so there is never one on disk
    /// to find and nothing to batch.
    #[cfg(not(unix))]
    async fn materialized_variants(&self, digests: &[DigestInfo]) -> Vec<Option<OsString>> {
        vec![None; digests.len()]
    }

    /// Returns the path to a private, read-only **executable** (0o555) copy of
    /// the blob for `digest`, creating it at most once. Callers **hardlink**
    /// the returned path into action input trees instead of copying the
    /// executable per action.
    ///
    /// Why this exists: a CAS blob is stored read-only **0o444** and shared
    /// across actions by hardlink, so it cannot carry the executable bit and
    /// must never be `chmod`'d (that mutates the shared inode — the #2347
    /// corruption class). Materializing an executable input therefore needs a
    /// separate 0o555 inode. Doing that copy *per action* opens a writable fd
    /// in the worker's hot path; under fork-heavy concurrency a child can
    /// inherit that fd and a concurrent `execve` of the executable then fails
    /// with `ETXTBSY` ("Text file busy", os error 26). Creating the 0o555 inode
    /// **once** — writer fd fsync'd and closed, then atomically renamed into
    /// place before the inode is ever hardlinked or executed — and hardlinking
    /// it thereafter keeps the per-action path hardlink-only.
    ///
    /// Lifetime: the result is a bare path, not a handle, and the variant is
    /// deleted by a removal of the digest's entry (see
    /// `ExecutableVariantRemover`). A caller that holds a lease on the
    /// digest ([`Self::lease_digest`]) from before this call until it has
    /// hardlinked the path, as a worker does for its action's inputs, finds
    /// the variant still there despite automatic eviction and despite the
    /// pending cleanup of an older removal: a path is returned only while a
    /// generation of the digest is resident, and the lease keeps that
    /// generation from being evicted. A digest that is leased but not resident
    /// gets an error, not its existing variant, so the caller populates it
    /// first. Nothing protects an unleased caller, an explicit removal of the
    /// leased entry (such as dropping an entry whose file went missing), or
    /// deletion from outside the store.
    #[cfg(unix)]
    pub async fn get_executable_hardlink_source(
        &self,
        digest: &DigestInfo,
    ) -> Result<OsString, Error> {
        if !self.executable_variants_enabled {
            return Err(make_err!(
                Code::FailedPrecondition,
                "Executable hardlink sources are disabled because the startup wipe could not safely clear the read-only executable directory"
            ));
        }
        let variant_path = self.executable_variant_path(digest);

        // Fast path: the variant already exists, so the caller can hardlink it
        // with no writable fd anywhere in sight. Unless the digest is not
        // resident or someone holds the lease: either may mean a removal is
        // about to delete the file.
        if self.variants_servable_without_lease(core::slice::from_ref(digest))[0]
            && fs::metadata(&variant_path).await.is_ok()
        {
            return Ok(variant_path);
        }

        // Resolve the on-disk CAS blob (0o444) to copy from before taking the
        // lease: resolving may reap an expired entry, whose removal callback
        // takes this digest's lease. Must be present in this tier; callers
        // populate the fast store first.
        let file_entry = self
            .get_file_entry_for_digest(digest)
            .await
            .err_tip(|| "Resolving CAS blob for executable variant")?;

        // Single-flight: exactly one task ever opens a writable fd for this
        // variant. Without this, a cold-cache burst of concurrent actions
        // would each open a writer for the same executable — the very window
        // ETXTBSY exploits.
        let lease = ExecutableVariantLease::acquire(&self.executable_locks, *digest).await;

        // Re-check: another task may have constructed it while we waited, and
        // no removal can be deleting it while we hold the lease.
        if fs::metadata(&variant_path).await.is_ok() {
            return Ok(variant_path);
        }

        let result = self
            .create_executable_variant(&file_entry, &variant_path, lease)
            .await;

        // The digest may have been removed mid-copy. Its removal callback then
        // deletes the variant once this copy releases the lease, or, if it ran
        // before the copy took the lease, left nothing to delete and this
        // variant is untracked until the digest's next removal. A caller that
        // keeps its inputs leased cannot see either: leased entries are not
        // evicted.
        if result.is_ok() && self.evicting_map.get(&digest.into()).await.is_none() {
            warn!(
                %digest,
                ?variant_path,
                "Digest removed while materializing its executable variant"
            );
        }

        result.map(|()| variant_path)
    }

    /// Non-unix has no executable bit and no `ETXTBSY`, so just hardlink the
    /// CAS blob directly.
    #[cfg(not(unix))]
    pub async fn get_executable_hardlink_source(
        &self,
        digest: &DigestInfo,
    ) -> Result<OsString, Error> {
        let file_entry = self.get_file_entry_for_digest(digest).await?;
        file_entry
            .get_file_path_locked(|p| async move { Ok(p) })
            .await
    }

    /// Materializes the 0o555 executable variant of `file_entry`'s blob, under
    /// `lease`, which the blocking copy holds until it has finished.
    #[cfg(unix)]
    async fn create_executable_variant(
        &self,
        file_entry: &Fe,
        variant_path: &OsStr,
        lease: ExecutableVariantLease,
    ) -> Result<(), Error> {
        let src_path = file_entry
            .get_file_path_locked(|p| async move { Ok(p) })
            .await?;

        let variant_owned = variant_path.to_os_string();
        let rename_fn = self.rename_fn;

        // All of this is blocking std::fs; run it off the async runtime. The
        // writable fd is fully closed before the `rename` publishes the inode,
        // so no reachable hardlink of the variant ever has an open writer.
        // This task, not its possibly-cancelled caller, owns the lease and the
        // temporary file: both are released when the copy has really ended.
        spawn_blocking!(
            "filesystem_store_executable_variant",
            move || -> Result<(), Error> {
                use std::os::unix::fs::PermissionsExt;
                let _lease = lease;
                let mut src = std::fs::File::open(&src_path).map_err(|e| {
                    make_err!(Code::Internal, "executable-variant copy failed: {e:?}")
                })?;
                let (mut temp, mut f) =
                    ExecutableVariantTemp::create(&variant_owned).map_err(|e| {
                        make_err!(Code::Internal, "executable-variant copy failed: {e:?}")
                    })?;
                std::io::copy(&mut src, &mut f).map_err(|e| {
                    make_err!(Code::Internal, "executable-variant copy failed: {e:?}")
                })?;
                drop(src);
                f.set_permissions(std::fs::Permissions::from_mode(0o555))
                    .map_err(|e| {
                        make_err!(
                            Code::Internal,
                            "executable-variant chmod 0o555 failed: {e:?}"
                        )
                    })?;
                // Belt-and-suspenders flush before publish. The `.exec`
                // directory is cleared before executable variants are enabled,
                // so durability is not needed across process restarts. The
                // flush still ensures a published variant is complete for the
                // current process. Non-macOS keeps the prior `sync_all`
                // behavior; macOS uses plain `fsync(2)` (data handed to the
                // device, no multi-ms `F_FULLFSYNC` device-cache drain), which
                // covers kernel panics — the next writable startup wipe covers
                // power loss.
                #[cfg(target_os = "macos")]
                let flush_result = {
                    use std::os::fd::AsRawFd;
                    loop {
                        if unsafe { libc::fsync(f.as_raw_fd()) } == 0 {
                            break Ok(());
                        }
                        let err = std::io::Error::last_os_error();
                        if err.kind() != std::io::ErrorKind::Interrupted {
                            break Err(err);
                        }
                    }
                };
                #[cfg(not(target_os = "macos"))]
                let flush_result = f.sync_all();
                flush_result
                    .map_err(|e| make_err!(Code::Internal, "executable-variant fsync: {e:?}"))?;
                drop(f);
                rename_fn(&temp.path, &variant_owned).map_err(|e| {
                    make_err!(Code::Internal, "executable-variant rename failed: {e:?}")
                })?;
                temp.published = true;
                Ok(())
            }
        )
        .await
        .err_tip(|| "executable-variant spawn_blocking join failed")?
    }

    pub async fn get_file_entry_for_digest(&self, digest: &DigestInfo) -> Result<Arc<Fe>, Error> {
        // Zero-digest blobs have no backing file on disk (FilesystemStore
        // never persists zero-byte content). The previous implementation
        // returned a synthetic FileEntry whose content_path did not exist,
        // which downstream callers would then try to hard_link from,
        // silently producing missing or empty output files in worker
        // execution directories. Return NotFound so callers are forced to
        // take the explicit zero-digest path (e.g. fs::create_file).
        if is_zero_digest(digest) {
            return Err(make_err!(
                Code::NotFound,
                "{digest} is a zero-digest; FilesystemStore does not persist zero-byte files. \
                 Callers must materialise empty files directly rather than going through get_file_entry_for_digest."
            ));
        }
        self.evicting_map
            .get(&digest.into())
            .await
            .ok_or_else(|| make_err!(Code::NotFound, "{digest} not found in filesystem store. This may indicate the file was evicted due to cache pressure. Consider increasing 'max_bytes' in your filesystem store's eviction_policy configuration."))
    }

    async fn update_file(
        self: Pin<&Self>,
        entry: Fe,
        mut temp_file: FileSlot,
        final_key: StoreKey<'static>,
        mut reader: DropCloserReadHalf,
        upload_size: UploadSizeInfo,
    ) -> Result<u64, Error> {
        let mut data_size = 0;
        loop {
            let mut data = reader
                .recv()
                .await
                .err_tip(|| "Failed to receive data in filesystem store")?;
            let data_len = data.len();
            if data_len == 0 {
                break; // EOF.
            }
            temp_file
                .write_all_buf(&mut data)
                .await
                .err_tip(|| "Failed to write data into filesystem store")?;
            data_size += data_len as u64;
        }

        upload_size.validate_size(data_size)?;
        self.finalize_temp_file(entry, temp_file, data_size, final_key)
            .await?;
        Ok(data_size)
    }

    /// Makes a completely written temp file durable and publishes it as
    /// `final_key`.
    ///
    /// The `max_concurrent_writes` permit bounds the flush, the fsync and the
    /// publication (the rename and the map insertion), so it must be held
    /// until they have really finished. Once the permit is granted, it moves
    /// with the file and the entry into one detached task that does all of
    /// that work and releases the permit only when it ends. Dropping the
    /// caller can therefore never release the permit while its write is
    /// still running: like the rename on its own before (see issue #495), the
    /// write finishes, or fails and cleans up, without the caller.
    ///
    /// While it holds the permit, that task must not wait for an open-file
    /// permit, nor for a lock that is held by a task waiting for one: writers
    /// queued for the write permit keep their temp files, and with them
    /// open-file permits, until they get it. So the duplicate check, which
    /// opens the stored file and takes its entry's lock, runs before the
    /// write permit is requested, and the cleanup of entries displaced by the
    /// insertion runs after it is released.
    async fn finalize_temp_file(
        self: Pin<&Self>,
        mut entry: Fe,
        mut temp_file: FileSlot,
        data_size: u64,
        final_key: StoreKey<'static>,
    ) -> Result<(), Error> {
        let store = self
            .get_arc()
            .err_tip(|| "Filesystem store dropped while finalizing a write")?;
        *entry.data_size_mut() = data_size;
        let entry = Arc::new(entry);
        // A blob identical to the stored one needs no second copy. The check
        // can remove the stored entry, when it has expired or its file is
        // gone, and then has to retire it, so like the publication it runs
        // in a detached task that the caller cannot cancel (see issue #495).
        // It holds no write permit, so it may wait for open-file permits.
        let (is_duplicate, mut temp_file) =
            background_spawn!("filesystem_store_check_duplicate", {
                let evicting_map = self.evicting_map.clone();
                let (final_key, entry) = (final_key.clone(), entry.clone());
                async move {
                    let is_duplicate = check_duplicate_files_with(
                        &evicting_map,
                        &final_key,
                        &entry,
                        Some(&mut temp_file),
                    )
                    .await;
                    (is_duplicate, temp_file)
                }
            })
            .await
            .err_tip(|| "Failed to join duplicate check task in filesystem store")?;
        if is_duplicate? {
            return Ok(());
        }
        // Waiting for the permit may be cancelled: no protected work has
        // started yet.
        let permit = match &self.write_semaphore {
            Some(sem) => Some(Arc::clone(sem).acquire_owned().await.map_err(|err| {
                Error::from_std_err(Code::Internal, &err).append("Write semaphore closed")
            })?),
            None => None,
        };
        background_spawn!("filesystem_store_finalize_file", async move {
            let published: Result<AfterPublication<Fe>, Error> = async {
                // tokio defers write errors to the next write or flush call, so without an explicit flush
                // the final write's failure is silently swallowed by `sync_all`, and a truncated file would
                // be renamed into the content path.
                temp_file
                    .flush()
                    .await
                    .err_tip(|| "Failed to flush in filesystem store")?;
                store
                    .flush_durably(&temp_file)
                    .await
                    .err_tip(|| "Failed to sync in filesystem store")?;
                // Lock the blob down as read-only before publishing it, as
                // `emplace_file` does, but through the open file: setting
                // permissions by path would take another open-file permit.
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if let Err(err) = temp_file
                        .as_ref()
                        .set_permissions(std::fs::Permissions::from_mode(0o444))
                        .await
                    {
                        warn!(?err, ?temp_file, "Failed to set CAS blob read-only");
                    }
                }

                if store.evict_page_cache {
                    temp_file.advise_dontneed();
                }
                trace!(?temp_file, "Dropping file to finalize_temp_file");
                drop(temp_file);

                publish_file(&store.evicting_map, store.rename_fn, final_key, entry).await
            }
            .await;
            drop(permit);
            published?.run().await;
            Ok(())
        })
        .await
        .err_tip(|| "Failed to join finalize task in filesystem store")?
    }

    async fn emplace_file(&self, key: StoreKey<'static>, entry: Arc<Fe>) -> Result<(), Error> {
        let evicting_map = self.evicting_map.clone();
        let rename_fn = self.rename_fn;

        // We need to guarantee that this will get to the end even if the parent future is dropped.
        // See: https://github.com/TraceMachina/nativelink/issues/495
        background_spawn!("filesystem_store_emplace_file", async move {
            // Sometimes we get files to emplace that are identical to the existing files
            // Due to the evict/remove/replace cycle taking some amount of time, we actually
            // want to drop these
            if check_duplicate_files(&evicting_map, &key, &entry).await? {
                return Ok(());
            }

            // Lock the blob down as read-only *before* it lands at its final
            // content path, so every hardlink of it (the worker directory cache
            // and `download_to_directory`) inherits an immutable, read-only
            // inode. This is what preserves the input-tree hermeticity contract
            // (actions cannot mutate their inputs) without a per-materialization
            // chmod walk, and it means callers must never `chmod` a hardlinked
            // blob — anything needing a different mode (e.g. an executable's +x
            // bit) must take a private copy. Content-addressed blobs are
            // write-once and replaced by `rename` rather than in-place writes,
            // and unlink only needs the *parent directory* to be writable, so a
            // read-only file mode is safe for both overwrite-by-rename and
            // eviction. Best-effort: a failure here (e.g. the temp file was
            // already evicted out from under us) must not abort the emplace.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let from_path = entry
                    .get_encoded_file_path()
                    .read()
                    .await
                    .get_file_path()
                    .into_owned();
                if let Err(err) =
                    fs::set_permissions(&from_path, std::fs::Permissions::from_mode(0o444)).await
                {
                    warn!(?err, ?from_path, "Failed to set CAS blob read-only");
                }
            }

            publish_file(&evicting_map, rename_fn, key, entry)
                .await?
                .run()
                .await;
            Ok(())
        })
        .await
        .err_tip(|| "Failed to create spawn in filesystem store update_file")?
    }

    /// Makes `file`'s contents durable with `sync_all` semantics.
    ///
    /// On macOS the naive equivalent (`fcntl(F_FULLFSYNC)` per file) is a
    /// full device-cache flush that serializes at the device and dominates
    /// many-small-blob uploads, so the flush is split into the cheap
    /// per-file part (`fsync(2)`: push this file's pages to the device)
    /// and a device-wide cache drain shared with every concurrent upload
    /// via [`FlushCoalescer`]. Other platforms get their journal's group
    /// commit for free and keep plain `sync_all`.
    async fn flush_durably(&self, file: &FileSlot) -> Result<(), Error> {
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            let Some(flush_coalescer) = &self.flush_coalescer else {
                return file.as_ref().sync_all().await.map_err(Into::into);
            };
            // Dup the handle so the blocking task owns its own fd: if
            // this future is cancelled mid-await, the temp file's fd can
            // close and be reused, and an already-started blocking task
            // must never fsync an unrelated descriptor.
            let file_dup = file
                .as_ref()
                .try_clone()
                .await
                .err_tip(|| "Failed to dup fd for fsync in filesystem store")?
                .into_std()
                .await;
            // Deliberately NOT via `fs::call_with_permit`: the caller's
            // `FileSlot` already holds an open-file permit, so requiring a
            // second one here deadlocks when the semaphore is drained (the
            // exact scenario the #2051 regression test pins at one
            // permit). Concurrency is still bounded: every in-flight fsync
            // belongs to an upload holding a `FileSlot` permit.
            spawn_blocking!("filesystem_store_fsync", move || {
                loop {
                    if unsafe { libc::fsync(file_dup.as_raw_fd()) } == 0 {
                        return Ok(());
                    }
                    let err = std::io::Error::last_os_error();
                    // std's sync_all retries EINTR (cvt_r); match it.
                    if err.kind() != std::io::ErrorKind::Interrupted {
                        return Err(Error::from(err).append("fsync failed in filesystem store"));
                    }
                }
            })
            .await
            .err_tip(|| "Failed to join fsync task in filesystem store")??;
            flush_coalescer.commit().await
        }
        #[cfg(not(target_os = "macos"))]
        {
            file.as_ref().sync_all().await.map_err(Into::into)
        }
    }

    pub fn get_eviction_snapshot(&self) -> EvictionSnapshot {
        self.evicting_map.get_snapshot()
    }

    // Only for tests, so we can run check_duplicate_files
    pub fn get_evicting_map(&self) -> Arc<FsEvictingMap<'static, Fe>> {
        self.evicting_map.clone()
    }

    /// Takes this store's retirement admission (see
    /// [`SharedContext::retire_blocking`]) until the permit is dropped, so a
    /// test can stand in for a slow earlier retirement without making one.
    pub async fn hold_retirement_admission_for_test(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.shared_context
            .retirement
            .clone()
            .acquire_owned()
            .await
            .expect("retirement admission is never closed")
    }

    /// Callers holding or queued for `digest`'s executable-variant lease, so a
    /// test can tell when a caller is waiting behind another.
    #[cfg(unix)]
    pub fn executable_variant_lease_claims_for_test(&self, digest: &DigestInfo) -> usize {
        self.executable_locks
            .lock()
            .expect("executable_locks poisoned")
            .get(digest)
            .map_or(0, |lock| Arc::strong_count(lock) - 1)
    }

    /// Returns the number of device-wide durability barriers issued by this
    /// store's macOS flush coalescer. Exposed so the integration test can pin
    /// the batching behavior rather than only checking data correctness.
    #[cfg(target_os = "macos")]
    pub fn full_flush_count_for_test(&self) -> Option<u64> {
        self.flush_coalescer
            .as_ref()
            .map(|coalescer| coalescer.full_flush_count.load(Ordering::Relaxed))
    }

    // Separated out so tests can use this
    pub async fn make_temp_file(
        &self,
        temp_key: StoreKey<'static>,
    ) -> Result<(Fe, FileSlot, OsString), Error> {
        let generation = self.get_and_update_generation()?;

        Fe::make_and_open_file(
            self.block_size,
            generation,
            EncodedFilePath {
                shared_context: self.shared_context.clone(),
                path_type: PathType::Temp,
                key: temp_key,
                generation,
                version: Version::V2,
            },
        )
        .await
    }
}

#[async_trait]
impl<Fe: FileEntry> StoreDriver for FilesystemStore<Fe> {
    async fn post_init(self: Arc<Self>) -> Result<(), Error> {
        Ok(())
    }

    async fn remove(self: Pin<&Self>, key: StoreKey<'_>) -> Result<bool, Error> {
        Ok(self.evicting_map.remove(&key.into_owned()).await)
    }

    async fn has_with_results(
        self: Pin<&Self>,
        keys: &[StoreKey<'_>],
        results: &mut [Option<u64>],
    ) -> Result<(), Error> {
        let own_keys = keys
            .iter()
            .map(|sk| sk.borrow().into_owned())
            .collect::<Vec<_>>();
        self.evicting_map
            .sizes_for_keys(own_keys.iter(), results, false /* peek */)
            .await;
        // We need to do a special pass to ensure our zero files exist.
        // If our results failed and the result was a zero file, we need to
        // create the file by spec.
        for (key, result) in keys.iter().zip(results.iter_mut()) {
            if result.is_some() || !is_zero_digest(key.borrow()) {
                continue;
            }
            let (mut tx, rx) = make_buf_channel_pair();
            let send_eof_result = tx.send_eof();
            self.update(key.borrow(), rx, UploadSizeInfo::ExactSize(0))
                .await
                .err_tip(|| format!("Failed to create zero file for key {}", key.as_str()))
                .merge(
                    send_eof_result
                        .err_tip(|| "Failed to send zero file EOF in filesystem store has"),
                )?;

            *result = Some(0);
        }
        Ok(())
    }

    async fn update(
        self: Pin<&Self>,
        key: StoreKey<'_>,
        mut reader: DropCloserReadHalf,
        upload_size: UploadSizeInfo,
    ) -> Result<u64, Error> {
        if is_zero_digest(key.borrow()) {
            let size = reader.drain().await?;
            UploadSizeInfo::ExactSize(0).validate_size(size)?;
            upload_size.validate_size(size)?;
            return Ok(0);
        }

        let temp_key = make_temp_key(&key);

        // There's a possibility of deadlock here where we take all of the
        // file semaphores with make_and_open_file and the semaphores for
        // whatever is populating reader is exhasted on the threads that
        // have the FileSlots and not on those which can't.  To work around
        // this we don't take the FileSlot until there's something on the
        // reader available to know that the populator is active.
        reader.peek().await?;

        let (entry, temp_file, temp_full_path) = self.make_temp_file(temp_key).await?;

        self.update_file(entry, temp_file, key.into_owned(), reader, upload_size)
            .await
            .err_tip(|| {
                format!(
                    "While processing with temp file {}",
                    temp_full_path.display()
                )
            })
    }

    fn optimized_for(&self, optimization: StoreOptimizations) -> bool {
        matches!(
            optimization,
            StoreOptimizations::FileUpdates | StoreOptimizations::SubscribesToUpdateOneshot
        )
    }

    async fn update_oneshot(self: Pin<&Self>, key: StoreKey<'_>, data: Bytes) -> Result<(), Error> {
        if is_zero_digest(key.borrow()) {
            UploadSizeInfo::ExactSize(0).validate_size(data.len() as u64)?;
            return Ok(());
        }

        let temp_key = make_temp_key(&key);
        let (entry, mut temp_file, temp_full_path) = self
            .make_temp_file(temp_key)
            .await
            .err_tip(|| "Failed to create temp file in filesystem store update_oneshot")?;

        // Write directly without channel overhead
        if !data.is_empty() {
            temp_file
                .write_all(&data)
                .await
                .err_tip(|| format!("Failed to write data to {}", temp_full_path.display()))?;
        }

        self.finalize_temp_file(entry, temp_file, data.len() as u64, key.into_owned())
            .await
            .err_tip(|| "In filesystem store update_oneshot")
    }

    async fn update_with_whole_file(
        self: Pin<&Self>,
        key: StoreKey<'_>,
        path: OsString,
        mut file: FileSlot,
        upload_size: UploadSizeInfo,
    ) -> Result<(u64, Option<FileSlot>), Error> {
        file.flush()
            .await
            .err_tip(|| "Flushing whole-file upload before checking its size")?;
        let file_size = file
            .as_ref()
            .metadata()
            .await
            .err_tip(|| format!("While reading metadata for {}", path.display()))?
            .len();
        upload_size.validate_size(file_size)?;
        if is_zero_digest(key.borrow()) {
            UploadSizeInfo::ExactSize(0).validate_size(file_size)?;
            // Empty CAS blobs are implicit; ordinary empty values still need a file.
            return Ok((0, None));
        }
        let generation = self.get_and_update_generation()?;
        let entry = Fe::create(
            file_size,
            self.block_size,
            generation,
            RwLock::new(EncodedFilePath {
                shared_context: self.shared_context.clone(),
                path_type: PathType::Custom(path),
                key: key.borrow().into_owned(),
                generation,
                version: Version::V2,
            }),
        );
        // We are done with the file, if we hold a reference to the file here, it could
        // result in a deadlock if `emplace_file()` also needs file descriptors.
        trace!(?file, "Dropping file to to update_with_whole_file");
        if self.evict_page_cache {
            file.advise_dontneed();
        }
        drop(file);
        self.emplace_file(key.into_owned(), Arc::new(entry))
            .await
            .err_tip(|| "Could not move file into store in upload_file_to_store, maybe dest is on different volume?")?;
        return Ok((file_size, None));
    }

    async fn get_part(
        self: Pin<&Self>,
        key: StoreKey<'_>,
        writer: &mut DropCloserWriteHalf,
        offset: u64,
        length: Option<u64>,
    ) -> Result<(), Error> {
        if is_zero_digest(key.borrow()) {
            self.has(key.borrow())
                .await
                .err_tip(|| "Failed to check if zero digest exists in filesystem store")?;
            writer
                .send_eof()
                .err_tip(|| "Failed to send zero EOF in filesystem store get_part")?;
            return Ok(());
        }
        let owned_key = key.into_owned();
        let entry = self.evicting_map.get(&owned_key).await.ok_or_else(|| {
            make_err!(
                Code::NotFound,
                "{} not found in filesystem store here",
                owned_key.as_str()
            )
        })?;
        let read_limit = length.unwrap_or(u64::MAX);
        let mut temp_file = match entry.read_file_part(offset, read_limit).await {
            Ok(file) => file,
            Err(err) => {
                // If the file is not found, we need to remove it from the eviction map.
                if err.code == Code::NotFound {
                    // Map said the file was present but `open()` hit ENOENT.
                    // Self-heals: we remove the stale entry below and a
                    // fast/slow caller re-populates from the slow store, so
                    // this is a recoverable warn, not a fatal error.
                    warn!(
                        ?err,
                        key = ?owned_key,
                        "Filesystem store map/disk divergence: removing entry; reader will fall through to slow store",
                    );
                    self.evicting_map
                        .remove_if(&owned_key, |map_entry| Arc::ptr_eq(map_entry, &entry))
                        .await;
                }
                return Err(err);
            }
        };

        loop {
            let mut buf = BytesMut::with_capacity(self.read_buffer_size);
            temp_file
                .read_buf(&mut buf)
                .await
                .err_tip(|| "Failed to read data in filesystem store")?;
            if buf.is_empty() {
                break; // EOF.
            }
            writer
                .send(buf.freeze())
                .await
                .err_tip(|| "Failed to send chunk in filesystem store get_part")?;
        }
        if self.evict_page_cache {
            temp_file.get_ref().advise_dontneed();
        }
        writer
            .send_eof()
            .err_tip(|| "Filed to send EOF in filesystem store get_part")?;

        Ok(())
    }

    fn inner_store(&self, _digest: Option<StoreKey>) -> &dyn StoreDriver {
        self
    }

    fn as_any<'a>(&'a self) -> &'a (dyn core::any::Any + Sync + Send + 'static) {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn core::any::Any + Sync + Send + 'static> {
        self
    }

    fn register_health(self: Arc<Self>, registry: &mut HealthRegistryBuilder) {
        registry.register_indicator(self);
    }

    fn enable_cache_size_metrics(&self, attrs: &[opentelemetry::KeyValue]) -> bool {
        self.evicting_map.enable_cache_size_metrics(attrs.to_vec());
        true
    }

    fn register_remove_callback(self: Arc<Self>, callback: RemoveCallback) -> Result<(), Error> {
        self.evicting_map
            .add_remove_callback(RemoveCallbackHolder::new(callback));
        Ok(())
    }
}

#[async_trait]
impl<Fe: FileEntry> HealthStatusIndicator for FilesystemStore<Fe> {
    fn get_name(&self) -> &'static str {
        "FilesystemStore"
    }

    /// Lightweight probe: `stat()` the `content_path` directory. No
    /// write-semaphore / eviction-map contention with production
    /// traffic, and bounded so a hung NFS / EBS mount can't wedge the
    /// indicator.
    async fn check_health(&self, _namespace: Cow<'static, str>) -> HealthStatus {
        const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

        let content_path = &self.shared_context.content_path;
        let stat = tokio::fs::metadata(&content_path);
        match timeout(HEALTH_PROBE_TIMEOUT, stat).await {
            Ok(Ok(meta)) if meta.is_dir() => {
                HealthStatus::new_ok(self, "FilesystemStore::check_health: ok".into())
            }
            Ok(Ok(_)) => HealthStatus::new_failed(
                self,
                format!(
                    "FilesystemStore::check_health: content_path {content_path} is not a directory"
                )
                .into(),
            ),
            Ok(Err(e)) => {
                warn!(
                    ?e,
                    %content_path,
                    "FilesystemStore::check_health: stat errored",
                );
                HealthStatus::new_failed(
                    self,
                    format!("FilesystemStore::check_health: stat errored: {e}").into(),
                )
            }
            Err(_) => {
                warn!(
                    %content_path,
                    timeout_secs = HEALTH_PROBE_TIMEOUT.as_secs(),
                    "FilesystemStore::check_health: stat timed out",
                );
                HealthStatus::Timeout {
                    struct_name: self.struct_name(),
                }
            }
        }
    }
}

#[cfg(all(test, unix))]
mod executable_variant_lease_tests {
    use core::pin::pin;
    use std::sync::Arc;

    use futures::executor::block_on;
    use futures::poll;
    use nativelink_util::common::DigestInfo;

    use super::{ExecutableLocks, ExecutableVariantLease};

    fn holders(locks: &ExecutableLocks, digest: &DigestInfo) -> Option<usize> {
        locks
            .lock()
            .unwrap()
            .get(digest)
            .map(|lock| Arc::strong_count(lock) - 1)
    }

    /// Callers that queued on a lock which its lease then retired, and callers
    /// arriving afterwards, must still get the digest's lease one at a time.
    /// Polling by hand makes every interleaving point explicit: no timing.
    #[test]
    fn retired_lock_waiters_and_newcomers_take_turns() {
        block_on(async {
            let locks = Arc::<ExecutableLocks>::default();
            let digest = DigestInfo::new([7; 32], 1);

            let first = ExecutableVariantLease::acquire(&locks, digest).await;
            let mut old_1 = pin!(ExecutableVariantLease::acquire(&locks, digest));
            let mut old_2 = pin!(ExecutableVariantLease::acquire(&locks, digest));
            assert!(poll!(&mut old_1).is_pending());
            assert!(poll!(&mut old_2).is_pending());
            // Both old waiters hold the first lock: it, plus two waiters.
            assert_eq!(holders(&locks, &digest), Some(3));

            // A waiter cancelled while queued leaves nothing behind: the entry
            // stays owned by the lease it waited for, which retires it.
            {
                let mut cancelled = pin!(ExecutableVariantLease::acquire(&locks, digest));
                assert!(poll!(&mut cancelled).is_pending());
                assert_eq!(holders(&locks, &digest), Some(4));
            }
            assert_eq!(holders(&locks, &digest), Some(3));

            // Retiring the lease removes its entry before releasing the lock.
            drop(first);
            assert_eq!(holders(&locks, &digest), None);

            // A newcomer installs and takes a fresh lock. The old waiters may
            // now take the retired lock, but must find it retired and queue on
            // the newcomer's lock instead of proceeding.
            let newcomer = ExecutableVariantLease::acquire(&locks, digest).await;
            assert!(poll!(&mut old_1).is_pending());
            assert!(poll!(&mut old_2).is_pending());
            assert_eq!(holders(&locks, &digest), Some(3));

            // Retiring the newcomer's lease lets exactly one old waiter in.
            drop(newcomer);
            let polled = (poll!(&mut old_1), poll!(&mut old_2));
            let (winner, mut loser) = match polled {
                (core::task::Poll::Ready(lease), core::task::Poll::Pending) => (lease, old_2),
                (core::task::Poll::Pending, core::task::Poll::Ready(lease)) => (lease, old_1),
                (one, two) => panic!(
                    "exactly one old waiter must win: {:?} {:?}",
                    one.is_ready(),
                    two.is_ready()
                ),
            };
            assert!(poll!(&mut loser).is_pending());
            assert_eq!(holders(&locks, &digest), Some(2));

            drop(winner);
            let last = loser.await;
            assert_eq!(holders(&locks, &digest), Some(1));
            drop(last);
            assert!(locks.lock().unwrap().is_empty(), "every lease retired");
        });
    }
}

#[cfg(test)]
mod retirement_admission_tests {
    use core::time::Duration;
    use std::sync::Arc;

    use nativelink_error::Error;
    use nativelink_macro::nativelink_test;
    use nativelink_util::background_spawn;
    use tokio::sync::oneshot;

    use super::SharedContext;

    /// A retirement operation that has started keeps the store's admission
    /// after its caller is cancelled, until the operation itself returns, so
    /// a cancelled caller never lets a second operation run alongside it.
    /// Adapted from the independent review's probe (`t_d05da3eb`, run7610).
    #[nativelink_test]
    async fn started_retirement_keeps_admission_after_caller_abort() -> Result<(), Error> {
        let ctx = Arc::new(SharedContext::new(String::new(), String::new()));
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let first = background_spawn!("first_retirement", {
            let ctx = ctx.clone();
            async move {
                ctx.retire_blocking(move || {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                })
                .await
            }
        });
        tokio::time::timeout(Duration::from_secs(5), entered_rx)
            .await
            .expect("the first operation never started")
            .unwrap();
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        assert_eq!(
            ctx.retirement.available_permits(),
            0,
            "the started operation keeps admission"
        );

        let (second_tx, mut second_rx) = oneshot::channel();
        let second = background_spawn!("second_retirement", {
            let ctx = ctx.clone();
            async move {
                ctx.retire_blocking(move || second_tx.send(()).unwrap())
                    .await
            }
        });
        for _ in 0..32 {
            tokio::task::yield_now().await;
        }
        assert!(
            matches!(
                second_rx.try_recv(),
                Err(oneshot::error::TryRecvError::Empty)
            ),
            "a second operation ran while the first was still running"
        );
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), second_rx)
            .await
            .expect("the second operation never ran")
            .unwrap();
        second.await.unwrap()?;
        assert_eq!(ctx.retirement.available_permits(), 1);
        Ok(())
    }
}
