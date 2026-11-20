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

use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use core::time::Duration;
use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::SystemTime;

use async_lock::RwLock;
use bytes::Bytes;
use futures::executor::block_on;
use futures::task::Poll;
use futures::{Future, FutureExt, poll};
use nativelink_config::stores::{EvictionPolicy, FilesystemSpec};
use nativelink_error::{Code, Error, ErrorContext, ResultExt, make_err};
use nativelink_macro::nativelink_test;
use nativelink_store::filesystem_store::{
    BlockingRetirementPeak, DIGEST_FOLDER_V1, DIGEST_FOLDER_V2, EncodedFilePath, FileEntry,
    FileEntryImpl, FileType, FilesystemStore, Generation, STR_FOLDER_V1, STR_FOLDER_V2,
    check_duplicate_files, key_and_generation_from_file_v2, make_temp_key,
};
use nativelink_util::buf_channel::make_buf_channel_pair;
use nativelink_util::common::{DigestInfo, fs, make_temp_path};
use nativelink_util::evicting_map::LenEntry;
use nativelink_util::health_utils::{HealthStatus, HealthStatusIndicator};
use nativelink_util::store_trait::{
    RemoveItemCallback, Store, StoreDriver, StoreKey, StoreLike, UploadSizeInfo,
};
use nativelink_util::{background_spawn, spawn, spawn_blocking};
use opentelemetry::context::{Context, FutureExt as OtelFutureExt};
use parking_lot::Mutex;
use pretty_assertions::assert_eq;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, Take};
use tokio::sync::{Barrier, Semaphore};
use tokio::time::sleep;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReadDirStream;
use tracing::{Instrument, debug, info};

const VALID_HASH: &str = "0123456789abcdef000000000000000000010000000000000123456789abcdef";

fn make_random_data(sz: usize) -> Vec<u8> {
    let mut value = vec![0u8; sz];
    let mut rng = SmallRng::seed_from_u64(1);
    rng.fill(&mut value[..]);
    value
}

trait FileEntryHooks {
    fn on_make_and_open(
        _encoded_file_path: &EncodedFilePath,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        core::future::ready(Ok(()))
    }
    fn on_unref<Fe: FileEntry>(_entry: &Fe) {}
    fn on_unref_async<Fe: FileEntry>(_entry: &Fe) -> impl Future<Output = ()> + Send {
        core::future::ready(())
    }
    fn on_read_error() -> impl Future<Output = ()> + Send {
        core::future::ready(())
    }
    fn on_drop<Fe: FileEntry>(_entry: &Fe) {}
}

struct TestFileEntry<Hooks: FileEntryHooks + 'static + Sync + Send> {
    inner: Option<FileEntryImpl>,
    _phantom: PhantomData<Hooks>,
}

impl<Hooks: FileEntryHooks + 'static + Sync + Send> Debug for TestFileEntry<Hooks> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("TestFileEntry")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<Hooks: FileEntryHooks + 'static + Sync + Send> FileEntry for TestFileEntry<Hooks> {
    fn create(
        data_size: u64,
        block_size: u64,
        generation: Generation,
        encoded_file_path: RwLock<EncodedFilePath>,
    ) -> Self {
        Self {
            inner: Some(FileEntryImpl::create(
                data_size,
                block_size,
                generation,
                encoded_file_path,
            )),
            _phantom: PhantomData,
        }
    }

    async fn make_and_open_file(
        block_size: u64,
        generation: Generation,
        encoded_file_path: EncodedFilePath,
    ) -> Result<(Self, fs::FileSlot, OsString), Error> {
        Hooks::on_make_and_open(&encoded_file_path).await?;
        let (inner, file_slot, path) =
            FileEntryImpl::make_and_open_file(block_size, generation, encoded_file_path).await?;
        Ok((
            Self {
                inner: Some(inner),
                _phantom: PhantomData,
            },
            file_slot,
            path,
        ))
    }

    fn data_size(&self) -> u64 {
        self.inner.as_ref().unwrap().data_size()
    }

    fn data_size_mut(&mut self) -> &mut u64 {
        self.inner.as_mut().unwrap().data_size_mut()
    }

    fn size_on_disk(&self) -> u64 {
        self.inner.as_ref().unwrap().size_on_disk()
    }

    fn get_encoded_file_path(&self) -> &RwLock<EncodedFilePath> {
        self.inner.as_ref().unwrap().get_encoded_file_path()
    }

    async fn try_unref(&self) -> bool {
        // Like `FileEntryImpl`: give up at once while a reader holds the
        // path, before running the hooks, so a hooked unref runs once. (A
        // reader taking the lock between this check and the inner attempt
        // makes the hooks run again in the detached unref.)
        if self.get_encoded_file_path().try_write().is_none() {
            return false;
        }
        Hooks::on_unref(self);
        Hooks::on_unref_async(self).await;
        self.inner.as_ref().unwrap().try_unref().await
    }

    fn generation(&self) -> Generation {
        self.inner.as_ref().unwrap().generation()
    }

    async fn read_file_part(&self, offset: u64, length: u64) -> Result<Take<fs::FileSlot>, Error> {
        let result = self
            .inner
            .as_ref()
            .unwrap()
            .read_file_part(offset, length)
            .await;
        if result.is_err() {
            Hooks::on_read_error().await;
        }
        result
    }

    async fn get_file_path_locked<
        T,
        Fut: Future<Output = Result<T, Error>> + Send,
        F: FnOnce(OsString) -> Fut + Send,
    >(
        &self,
        handler: F,
    ) -> Result<T, Error> {
        self.inner
            .as_ref()
            .unwrap()
            .get_file_path_locked(handler)
            .await
    }
}

impl<Hooks: FileEntryHooks + 'static + Sync + Send> LenEntry for TestFileEntry<Hooks> {
    fn len(&self) -> u64 {
        self.inner.as_ref().unwrap().len()
    }

    fn content_len(&self) -> u64 {
        self.inner.as_ref().unwrap().content_len()
    }

    fn is_empty(&self) -> bool {
        self.inner.as_ref().unwrap().is_empty()
    }

    async fn unref(&self) {
        Hooks::on_unref(self);
        Hooks::on_unref_async(self).await;
        self.inner.as_ref().unwrap().unref().await;
    }
}

impl<Hooks: FileEntryHooks + 'static + Sync + Send> Drop for TestFileEntry<Hooks> {
    fn drop(&mut self) {
        let mut inner = self.inner.take().unwrap();
        let shared_context = inner.get_shared_context_for_test();
        let current_context = Context::current();

        // We do this complicated bit here because tokio does not give a way to run a
        // command that will wait for all tasks and sub spawns to complete.
        // Sadly we need to rely on `active_drop_spawns` to hit zero to ensure that
        // all tasks have completed.
        let fut = async move {
            // Drop the FileEntryImpl in a controlled setting then wait for the
            // `active_drop_spawns` to hit zero.
            drop(inner);
            while shared_context.active_drop_spawns.load(Ordering::Acquire) > 0 {
                tokio::task::yield_now().await;
            }
        }
        .instrument(tracing::error_span!("test_file_entry_drop"))
        .with_context(current_context);

        #[expect(clippy::disallowed_methods, reason = "testing implementation")]
        let thread_handle = {
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                rt.block_on(fut);
            })
        };
        thread_handle.join().unwrap();
        // At this point we can guarantee our file drop spawn has completed.
        Hooks::on_drop(self);
    }
}

async fn read_file_contents(file_name: &OsStr) -> Result<Vec<u8>, Error> {
    let mut file = fs::open_file(file_name, 0, u64::MAX)
        .await
        .err_tip(|| format!("Failed to open file: {}", file_name.display()))?;
    let mut data = vec![];
    file.read_to_end(&mut data)
        .await
        .err_tip(|| "Error reading file to end")?;
    Ok(data)
}

async fn wait_for_no_open_files() -> Result<(), Error> {
    let mut counter = 0;
    while fs::get_open_files_for_test() != 0 {
        sleep(Duration::from_millis(1)).await;
        counter += 1;
        if counter > 1000 {
            return Err(make_err!(
                Code::Internal,
                "Timed out waiting all files to close"
            ));
        }
    }
    Ok(())
}

/// Find a file backing `key` in the content directory.
async fn find_content_file(
    content_path: &str,
    key: &StoreKey<'_>,
) -> Result<Option<OsString>, Error> {
    let (folder, prefix) = match key {
        StoreKey::Digest(digest) => (DIGEST_FOLDER_V2, format!("{digest}-")),
        StoreKey::Str(name) => (STR_FOLDER_V2, format!("{name}-")),
    };
    let (_permit, dir_handle) = fs::read_dir(format!("{content_path}/{folder}"))
        .await
        .err_tip(|| "Failed opening content directory")?
        .into_inner();
    let mut read_dir_stream = ReadDirStream::new(dir_handle);
    while let Some(dir_entry) = read_dir_stream.next().await {
        let dir_entry = dir_entry?;
        if dir_entry.file_name().to_string_lossy().starts_with(&prefix) {
            return Ok(Some(dir_entry.path().into_os_string()));
        }
    }
    Ok(None)
}

/// As [`find_content_file`], erroring when nothing backs `key`.
async fn content_file_path(content_path: &str, key: &StoreKey<'_>) -> Result<OsString, Error> {
    find_content_file(content_path, key)
        .await?
        .err_tip(|| format!("No content file for {key:?} under {content_path}"))
}

/// Whether the content directory holds a file for `key`.
async fn content_file_exists(content_path: &str, key: &StoreKey<'_>) -> Result<bool, Error> {
    Ok(find_content_file(content_path, key).await?.is_some())
}

/// Helper function to ensure there are no temporary or content files left.
async fn check_storage_dir_empty(storage_path: &str) -> Result<(), Error> {
    let (_permit, temp_dir_handle) = fs::read_dir(format!("{storage_path}/{DIGEST_FOLDER_V2}"))
        .await
        .err_tip(|| "Failed opening temp directory")?
        .into_inner();

    let mut read_dir_stream = ReadDirStream::new(temp_dir_handle);

    if let Some(temp_dir_entry) = read_dir_stream.next().await {
        let path = temp_dir_entry?.path();
        panic!(
            "No files should exist in temp directory, found: {}",
            path.display()
        );
    }

    let (_permit, temp_dir_handle) = fs::read_dir(format!("{storage_path}/{STR_FOLDER_V2}"))
        .await
        .err_tip(|| "Failed opening temp directory")?
        .into_inner();

    let mut read_dir_stream = ReadDirStream::new(temp_dir_handle);

    if let Some(temp_dir_entry) = read_dir_stream.next().await {
        let path = temp_dir_entry?.path();
        panic!(
            "No files should exist in temp directory, found: {}",
            path.display()
        );
    }
    Ok(())
}

async fn list_file_names(dir: &str) -> Result<Vec<String>, Error> {
    let (_permit, dir_handle) = fs::read_dir(dir)
        .await
        .err_tip(|| format!("Failed opening {dir}"))?
        .into_inner();
    let mut names = Vec::new();
    let mut read_dir_stream = ReadDirStream::new(dir_handle);
    while let Some(dir_entry) = read_dir_stream.next().await {
        names.push(dir_entry?.file_name().to_string_lossy().into_owned());
    }
    names.sort();
    Ok(names)
}

async fn assert_entries_readable(store: &FilesystemStore) -> Result<(), Error> {
    for (key, value) in LEGACY_STR_ENTRIES {
        let store_key = StoreKey::new_str(key);
        assert_eq!(
            store.has(store_key.borrow()).await?,
            Some(value.len() as u64),
            "string key {key} should be resident"
        );
        assert_eq!(
            store.get_part_unchunked(store_key, 0, None).await?,
            Bytes::from(value),
            "string key {key} should map to its own contents"
        );
    }
    Ok(())
}

const HASH1: &str = "0123456789abcdef000000000000000000010000000000000123456789abcdef";
const HASH2: &str = "0123456789abcdef000000000000000000020000000000000123456789abcdef";
const VALUE1: &str = "0123456789";
const VALUE2: &str = "9876543210";
const STRING_NAME: &str = "String_Filename";

#[nativelink_test]
async fn evict_page_cache_defaults_off_and_round_trips_when_on() -> Result<(), Error> {
    assert!(
        !FilesystemSpec::default().evict_page_cache,
        "evict_page_cache should default to off"
    );

    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let store = Store::new(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            eviction_policy: None,
            block_size: 1,
            evict_page_cache: true,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest, VALUE1.into()).await?;
    let content = store
        .get_part_unchunked(StoreKey::Digest(digest), 0, None)
        .await?;
    assert_eq!(content, VALUE1.as_bytes());
    Ok(())
}

#[nativelink_test]
async fn valid_results_after_shutdown_test() -> Result<(), Error> {
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    {
        let store = Store::new(
            FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                eviction_policy: None,
                block_size: 1,
                ..Default::default()
            })
            .await?,
        );
        // Insert dummy value into store.
        store.update_oneshot(digest, VALUE1.into()).await?;

        assert_eq!(
            store.has(digest).await,
            Ok(Some(VALUE1.len() as u64)),
            "Expected filesystem store to have hash: {}",
            HASH1
        );
    }
    {
        // With a new store ensure content is still readable (ie: restores from shutdown).
        let store = Box::pin(
            FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
                content_path,
                temp_path,
                eviction_policy: None,
                ..Default::default()
            })
            .await?,
        );

        let key = StoreKey::Digest(digest);

        let content = store.get_part_unchunked(key, 0, None).await?;
        assert_eq!(content, VALUE1.as_bytes());
    }

    Ok(())
}

#[nativelink_test]
async fn temp_files_get_deleted_on_replace_test() -> Result<(), Error> {
    static DELETES_FINISHED: AtomicU32 = AtomicU32::new(0);
    struct LocalHooks {}
    impl FileEntryHooks for LocalHooks {
        fn on_drop<Fe: FileEntry>(_file_entry: &Fe) {
            DELETES_FINISHED.fetch_add(1, Ordering::Relaxed);
        }
    }

    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Box::pin(
        FilesystemStore::<TestFileEntry<LocalHooks>>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: Some(EvictionPolicy {
                max_count: 3,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?,
    );

    store.update_oneshot(digest1, VALUE1.into()).await?;

    {
        // Check to ensure our file exists where it should and content matches.
        let content_file = content_file_path(&content_path, &StoreKey::Digest(digest1)).await?;
        let data = read_file_contents(&content_file).await?;
        assert_eq!(
            &data[..],
            VALUE1.as_bytes(),
            "Expected file content to match"
        );
    }

    // Replace content.
    store.update_oneshot(digest1, VALUE2.into()).await?;

    {
        // Replacing the content publishes a new generation, so re-discover it.
        let content_file = content_file_path(&content_path, &StoreKey::Digest(digest1)).await?;
        let data = read_file_contents(&content_file).await?;
        assert_eq!(
            &data[..],
            VALUE2.as_bytes(),
            "Expected file content to match"
        );
    }

    loop {
        if DELETES_FINISHED.load(Ordering::Relaxed) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }

    assert!(logs_contain(
        "Spawned a filesystem_delete_file current_active_drop_spawns=1"
    ));
    assert!(logs_contain(
        "Dropped a filesystem_delete_file current_active_drop_spawns=0"
    ));

    check_storage_dir_empty(&temp_path).await
}

// This test ensures that if a file is overridden and an open stream to the file already
// exists, the open stream will continue to work properly and when the stream is done the
// temporary file (of the object that was deleted) is cleaned up.
#[nativelink_test]
async fn file_continues_to_stream_on_content_replace_test() -> Result<(), Error> {
    static DELETES_FINISHED: AtomicU32 = AtomicU32::new(0);
    struct LocalHooks {}
    impl FileEntryHooks for LocalHooks {
        fn on_drop<Fe: FileEntry>(_file_entry: &Fe) {
            DELETES_FINISHED.fetch_add(1, Ordering::Relaxed);
        }
    }

    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Arc::new(
        FilesystemStore::<TestFileEntry<LocalHooks>>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: Some(EvictionPolicy {
                max_count: 3,
                ..Default::default()
            }),
            block_size: 1,
            read_buffer_size: 1,
            ..Default::default()
        })
        .await?,
    );

    // Insert data into store.
    store.update_oneshot(digest1, VALUE1.into()).await?;

    let (writer, mut reader) = make_buf_channel_pair();
    let store_clone = store.clone();
    let digest1_clone = digest1;
    background_spawn!(
        "file_continues_to_stream_on_content_replace_test_store_get",
        async move { store_clone.get(digest1_clone, writer).await.unwrap() },
    );

    {
        // Check to ensure our first byte has been received. The future should be stalled here.
        let first_byte = reader
            .consume(Some(1))
            .await
            .err_tip(|| "Error reading first byte")?;
        assert_eq!(
            first_byte[0],
            VALUE1.as_bytes()[0],
            "Expected first byte to match"
        );
    }

    // Replace content.
    store.update_oneshot(digest1, VALUE2.into()).await?;

    // The write queued the replaced entry's retirement; wait for it.
    store.retirement_settled_for_test().await;

    {
        // Now ensure we only have 1 file in our temp path - we know it is a digest.
        let (_permit, temp_dir_handle) = fs::read_dir(format!("{temp_path}/{DIGEST_FOLDER_V2}"))
            .await
            .err_tip(|| "Failed opening temp directory")?
            .into_inner();
        let mut read_dir_stream = ReadDirStream::new(temp_dir_handle);
        let mut num_files = 0;
        while let Some(temp_dir_entry) = read_dir_stream.next().await {
            num_files += 1;
            let path = temp_dir_entry?.path();
            let data = read_file_contents(path.as_os_str()).await?;
            assert_eq!(
                &data[..],
                VALUE1.as_bytes(),
                "Expected file content to match"
            );
        }
        assert_eq!(
            num_files, 1,
            "There should only be one file in the temp directory"
        );
    }

    let remaining_file_data = reader
        .consume(Some(1024))
        .await
        .err_tip(|| "Error reading remaining bytes")?;

    assert_eq!(
        &remaining_file_data,
        &VALUE1.as_bytes()[1..],
        "Expected file content to match"
    );

    loop {
        if DELETES_FINISHED.load(Ordering::Relaxed) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }

    // Now ensure our temp file was cleaned up.
    check_storage_dir_empty(&temp_path).await
}

// Eviction has a different code path than a file replacement, so we check that if a
// file is evicted and has an open stream on it, it will stay alive and eventually
// get deleted.
#[nativelink_test]
async fn file_gets_cleans_up_on_cache_eviction() -> Result<(), Error> {
    static DELETES_FINISHED: AtomicU32 = AtomicU32::new(0);
    struct LocalHooks {}
    impl FileEntryHooks for LocalHooks {
        fn on_drop<Fe: FileEntry>(_file_entry: &Fe) {
            DELETES_FINISHED.fetch_add(1, Ordering::Relaxed);
        }
    }

    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let digest2 = DigestInfo::try_new(HASH2, VALUE2.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Arc::new(
        FilesystemStore::<TestFileEntry<LocalHooks>>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: Some(EvictionPolicy {
                max_count: 1,
                ..Default::default()
            }),
            block_size: 1,
            read_buffer_size: 1,
            ..Default::default()
        })
        .await?,
    );

    // Insert data into store.
    store.update_oneshot(digest1, VALUE1.into()).await.unwrap();

    let mut reader = {
        let (writer, reader) = make_buf_channel_pair();
        let store_clone = store.clone();
        background_spawn!(
            "file_gets_cleans_up_on_cache_eviction_store_get",
            async move { store_clone.get(digest1, writer).await.unwrap() },
        );
        reader
    };
    // Ensure we have received 1 byte in our buffer. This will ensure we have a reference to
    // our file open.
    assert!(reader.peek().await.is_ok(), "Could not peek into reader");

    // Insert new content. This will evict the old item.
    store.update_oneshot(digest2, VALUE2.into()).await?;

    // The write queued the evicted entry's retirement; wait for it.
    store.retirement_settled_for_test().await;

    {
        // Now ensure we only have 1 file in our temp path - we know it is a digest.
        let (_permit, temp_dir_handle) = fs::read_dir(format!("{temp_path}/{DIGEST_FOLDER_V2}"))
            .await
            .err_tip(|| "Failed opening temp directory")?
            .into_inner();
        let mut read_dir_stream = ReadDirStream::new(temp_dir_handle);
        let mut num_files = 0;
        while let Some(temp_dir_entry) = read_dir_stream.next().await {
            num_files += 1;
            let path = temp_dir_entry?.path();
            let data = read_file_contents(path.as_os_str()).await?;
            assert_eq!(
                &data[..],
                VALUE1.as_bytes(),
                "Expected file content to match"
            );
        }
        assert_eq!(
            num_files, 1,
            "There should only be one file in the temp directory"
        );
    }

    let reader_data = reader
        .consume(Some(1024))
        .await
        .err_tip(|| "Error reading remaining bytes")?;

    assert_eq!(&reader_data, VALUE1, "Expected file content to match");

    loop {
        if DELETES_FINISHED.load(Ordering::Relaxed) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }

    // Now ensure our temp file was cleaned up.
    check_storage_dir_empty(&temp_path).await
}

// Test to ensure that if we are holding a reference to `FileEntry` and the contents are
// replaced, the `FileEntry` continues to use the old data.
// `FileEntry` file contents should be immutable for the lifetime of the object.
#[nativelink_test]
async fn digest_contents_replaced_continues_using_old_data() -> Result<(), Error> {
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            eviction_policy: None,
            ..Default::default()
        })
        .await?,
    );
    // Insert data into store.
    store.update_oneshot(digest, VALUE1.into()).await?;
    let file_entry = store.get_file_entry_for_digest(&digest).await?;
    {
        // The file contents should equal our initial data.
        let mut reader = file_entry.read_file_part(0, u64::MAX).await?;
        let mut file_contents = String::new();
        reader.read_to_string(&mut file_contents).await?;
        assert_eq!(file_contents, VALUE1);
    }

    // Now replace the data.
    store.update_oneshot(digest, VALUE2.into()).await?;

    {
        // The file contents still equal our old data.
        let mut reader = file_entry.read_file_part(0, u64::MAX).await?;
        let mut file_contents = String::new();
        reader.read_to_string(&mut file_contents).await?;
        assert_eq!(file_contents, VALUE1);
    }

    Ok(())
}

#[nativelink_test]
async fn eviction_on_insert_calls_unref_once() -> Result<(), Error> {
    const SMALL_VALUE: &str = "01";
    const BIG_VALUE: &str = "0123";

    static UNREFED_DIGESTS: LazyLock<Mutex<Vec<StoreKey<'static>>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    struct LocalHooks {}
    impl FileEntryHooks for LocalHooks {
        fn on_unref<Fe: FileEntry>(file_entry: &Fe) {
            block_on(file_entry.get_file_path_locked(move |path_str| async move {
                let path = Path::new(&path_str);
                let (digest, _generation) = key_and_generation_from_file_v2(
                    path.file_name().unwrap().to_str().unwrap(),
                    FileType::Digest,
                )
                .unwrap();
                UNREFED_DIGESTS.lock().push(digest.borrow().into_owned());
                Ok(())
            }))
            .unwrap();
        }
    }

    let small_digest = StoreKey::Digest(DigestInfo::try_new(HASH1, SMALL_VALUE.len())?);
    let big_digest = DigestInfo::try_new(HASH1, BIG_VALUE.len())?;

    let store = Box::pin(
        FilesystemStore::<TestFileEntry<LocalHooks>>::new(&FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            eviction_policy: Some(EvictionPolicy {
                max_bytes: 5,
                ..Default::default()
            }),
            block_size: 1,
            ..Default::default()
        })
        .await?,
    );
    // Insert data into store.
    store
        .update_oneshot(small_digest.borrow(), SMALL_VALUE.into())
        .await?;
    store.update_oneshot(big_digest, BIG_VALUE.into()).await?;
    // The write queued the evicted entry's retirement; wait for it.
    store.retirement_settled_for_test().await;

    {
        // Our first digest should have been unrefed exactly once.
        let unrefed_digests = UNREFED_DIGESTS.lock();
        assert_eq!(
            unrefed_digests.len(),
            1,
            "Expected exactly 1 unrefed digest"
        );
        assert_eq!(unrefed_digests[0], small_digest, "Expected digest to match");
    }

    Ok(())
}

#[nativelink_test]
async fn rename_on_insert_fails_due_to_filesystem_error_proper_cleanup_happens() -> Result<(), Error>
{
    const INITIAL_CONTENT: &str = "hello";

    async fn wait_for_temp_file<Fut: Future<Output = Result<(), Error>>, F: Fn() -> Fut>(
        temp_path: &str,
        yield_fn: F,
    ) -> Result<fs::DirEntry, Error> {
        loop {
            yield_fn().await?;
            // Now ensure we only have 1 file in our temp path - we know it is a digest.
            let (_permit, dir_handle) = fs::read_dir(format!("{temp_path}/{DIGEST_FOLDER_V2}"))
                .await?
                .into_inner();
            let mut read_dir_stream = ReadDirStream::new(dir_handle);
            if let Some(dir_entry) = read_dir_stream.next().await {
                assert!(
                    read_dir_stream.next().await.is_none(),
                    "There should only be one file in temp directory"
                );
                let dir_entry = dir_entry?;
                {
                    // Some filesystems won't sync automatically, so force it.
                    let file_handle = fs::open_file(dir_entry.path().into_os_string(), 0, u64::MAX)
                        .await
                        .err_tip(|| "Failed to open temp file")?;
                    // We don't care if it fails, this is only best attempt.
                    drop(file_handle.get_ref().as_ref().sync_all().await);
                }
                // Ensure we have written to the file too. This ensures we have an open file handle.
                // Failing to do this may result in the file existing, but the `update_fut` not actually
                // sending data to it yet.
                if dir_entry.metadata().await?.len() >= INITIAL_CONTENT.len() as u64 {
                    return Ok(dir_entry);
                }
            }
        }
        // Unreachable.
    }

    static FILE_DELETED_BARRIER: LazyLock<Arc<Barrier>> =
        LazyLock::new(|| Arc::new(Barrier::new(2)));

    struct LocalHooks {}
    impl FileEntryHooks for LocalHooks {
        fn on_drop<Fe: FileEntry>(_file_entry: &Fe) {
            background_spawn!(
                "rename_on_insert_fails_due_to_filesystem_error_proper_cleanup_happens_local_hooks_on_drop",
                FILE_DELETED_BARRIER.wait()
            );
        }
    }

    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;

    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Box::pin(
        FilesystemStore::<TestFileEntry<LocalHooks>>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: None,
            ..Default::default()
        })
        .await?,
    );

    let (mut tx, rx) = make_buf_channel_pair();
    let update_fut = Arc::new(async_lock::Mutex::new(store.update(
        digest,
        rx,
        UploadSizeInfo::MaxSize(100),
    )));
    // This will process as much of the future as it can before it needs to pause.
    // Our temp file will be created and opened and ready to have contents streamed
    // to it.
    assert_eq!(poll!(&mut *update_fut.lock().await)?, Poll::Pending);
    tx.send(INITIAL_CONTENT.into()).await?;

    // Now we extract that temp file that is generated.
    wait_for_temp_file(&temp_path, || {
        let update_fut_clone = update_fut.clone();
        async move {
            // This will ensure we yield to our future and other potential spawns.
            tokio::task::yield_now().await;
            assert_eq!(poll!(&mut *update_fut_clone.lock().await)?, Poll::Pending);
            Ok(())
        }
    })
    .await?;

    // Now make it impossible for the file to be moved into the final path.
    // This will trigger an error on `rename()`.
    fs::remove_dir_all(&content_path).await?;

    // Because send_eof() waits for shutdown of the rx side, we cannot just await in this thread.
    background_spawn!(
        "rename_on_insert_fails_due_to_filesystem_error_proper_cleanup_happens_send_eof",
        async move {
            tx.send_eof().unwrap();
        },
    );

    // Now finish waiting on update(). This should result in an error because we deleted our dest
    // folder.
    let update_result = &mut *update_fut.lock().await;
    assert!(
        update_result.await.is_err(),
        "Expected update to fail due to temp file being deleted before rename"
    );

    // Delete may happen on another thread, so wait for it.
    FILE_DELETED_BARRIER.wait().await;

    // Now it should have cleaned up its temp files.
    {
        check_storage_dir_empty(&temp_path).await?;
    }

    // Finally ensure that our entry is not in the store.
    assert_eq!(
        store.has(digest).await?,
        None,
        "Entry should not be in store"
    );
    Ok(())
}

#[nativelink_test]
async fn get_part_timeout_test() -> Result<(), Error> {
    let large_value = "x".repeat(1024);
    let digest = DigestInfo::try_new(HASH1, large_value.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Arc::new(
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                read_buffer_size: 1,
                ..Default::default()
            },
            |from, to| std::fs::rename(from, to),
        )
        .await?,
    );

    store
        .update_oneshot(digest, large_value.clone().into())
        .await?;

    let (writer, mut reader) = make_buf_channel_pair();
    let store_clone = store.clone();
    let digest_clone = digest;

    let _drop_guard = spawn!("get_part_timeout_test_get", async move {
        store_clone.get(digest_clone, writer).await
    });

    let file_data = reader
        .consume(Some(1024))
        .await
        .err_tip(|| "Error reading bytes")?;

    assert_eq!(
        &file_data,
        large_value.as_bytes(),
        "Expected file content to match"
    );

    Ok(())
}

#[nativelink_test]
async fn get_part_is_zero_digest() -> Result<(), Error> {
    let digest = DigestInfo::new(Sha256::new().finalize().into(), 0);
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Arc::new(
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                read_buffer_size: 1,
                ..Default::default()
            },
            |from, to| std::fs::rename(from, to),
        )
        .await?,
    );

    let store_clone = store.clone();
    let (mut writer, mut reader) = make_buf_channel_pair();

    let _drop_guard = spawn!("get_part_is_zero_digest_get_part", async move {
        drop(
            store_clone
                .get_part(digest, &mut writer, 0, None)
                .await
                .err_tip(|| "Failed to get_part"),
        );
    });

    let file_data = reader
        .consume(Some(1024))
        .await
        .err_tip(|| "Error reading bytes")?;

    let empty_bytes = Bytes::new();
    assert_eq!(&file_data, &empty_bytes, "Expected file content to match");

    Ok(())
}

#[nativelink_test]
async fn has_with_results_on_zero_digests() -> Result<(), Error> {
    let digest = DigestInfo::new(Sha256::new().finalize().into(), 0);
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Arc::new(
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                read_buffer_size: 1,
                ..Default::default()
            },
            |from, to| std::fs::rename(from, to),
        )
        .await?,
    );

    let keys = vec![digest.into()];
    let mut results = vec![None];
    drop(
        store
            .has_with_results(&keys, &mut results)
            .await
            .err_tip(|| "Failed to get_part"),
    );
    assert_eq!(results, vec![Some(0)]);

    check_storage_dir_empty(&content_path).await?;

    Ok(())
}

async fn wrap_update_zero_digest<F>(updater: F) -> Result<(), Error>
where
    F: AsyncFnOnce(DigestInfo, Arc<FilesystemStore>) -> Result<(), Error>,
{
    let digest = DigestInfo::new(Sha256::new().finalize().into(), 0);
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            read_buffer_size: 1,
            ..Default::default()
        },
        |from, to| std::fs::rename(from, to),
    )
    .await?;
    updater(digest, store).await?;
    check_storage_dir_empty(&content_path).await?;
    check_storage_dir_empty(&temp_path).await?;
    Ok(())
}

#[nativelink_test]
async fn update_whole_file_with_zero_digest() -> Result<(), Error> {
    wrap_update_zero_digest(async |digest, store| {
        let temp_file_dir = make_temp_path("update_with_zero_digest");
        std::fs::create_dir_all(&temp_file_dir)?;
        let temp_file_path = Path::new(&temp_file_dir).join("zero-length-file");
        std::fs::write(&temp_file_path, b"")
            .err_tip(|| format!("Writing to {temp_file_path:?}"))?;
        let file_slot = fs::open_file(&temp_file_path, 0, 0).await?.into_inner();
        store
            .update_with_whole_file(
                digest,
                temp_file_path.into(),
                file_slot,
                UploadSizeInfo::ExactSize(0),
            )
            .await?;
        Ok(())
    })
    .await
}

#[nativelink_test]
async fn update_oneshot_with_zero_digest() -> Result<(), Error> {
    wrap_update_zero_digest(async |digest, store| store.update_oneshot(digest, Bytes::new()).await)
        .await
}

#[nativelink_test]
async fn update_with_zero_digest() -> Result<(), Error> {
    wrap_update_zero_digest(async |digest, store| {
        let (mut writer, reader) = make_buf_channel_pair();
        writer.send_eof()?;
        store
            .update(digest, reader, UploadSizeInfo::ExactSize(0))
            .await
            .map(|_| ())
    })
    .await
}

#[nativelink_test]
async fn get_file_entry_for_zero_digest_returns_not_found() -> Result<(), Error> {
    // Regression test for: zero-digest blobs have no backing file on disk,
    // so `get_file_entry_for_digest` previously handed back a synthetic
    // FileEntry whose content_path did not exist. Downstream callers (the
    // worker output-materialisation path) would then try to hard_link from
    // that non-existent source, silently producing missing or empty output
    // files in worker exec dirs.
    //
    // After the fix, `get_file_entry_for_digest` returns NotFound for
    // zero digests so callers are forced to materialise empty files
    // directly (e.g. via `fs::create_file`). This complements the
    // DirectoryCache zero-byte short-circuit (input-fetch path); this
    // test covers the FilesystemStore API used by the output-upload /
    // hardlink path.
    let digest = DigestInfo::new(Sha256::new().finalize().into(), 0);
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            read_buffer_size: 1,
            ..Default::default()
        },
        |from, to| std::fs::rename(from, to),
    )
    .await?;

    let err = store
        .get_file_entry_for_digest(&digest)
        .await
        .expect_err("zero digest must not return a synthetic FileEntry");
    assert_eq!(err.code, Code::NotFound, "expected NotFound, got: {err:?}");
    Ok(())
}

/// Regression test for: <https://github.com/TraceMachina/nativelink/issues/495>.
#[nativelink_test(flavor = "multi_thread")]
async fn update_file_future_drops_before_rename() -> Result<(), Error> {
    // Mutex can be used to signal to the rename function to pause execution.
    static RENAME_REQUEST_PAUSE_MUX: async_lock::Mutex<()> = async_lock::Mutex::new(());
    // Boolean used to know if the rename function is currently paused.
    static RENAME_IS_PAUSED: AtomicBool = AtomicBool::new(false);

    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;

    let content_path = make_temp_path("content_path");
    let store = Arc::pin(
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: make_temp_path("temp_path"),
                eviction_policy: None,
                ..Default::default()
            },
            |from, to| {
                // If someone locked our mutex, it means we need to pause, so we
                // simply request a lock on the same mutex.
                if RENAME_REQUEST_PAUSE_MUX.try_lock().is_none() {
                    RENAME_IS_PAUSED.store(true, Ordering::Release);
                    while RENAME_REQUEST_PAUSE_MUX.try_lock().is_none() {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    RENAME_IS_PAUSED.store(false, Ordering::Release);
                }
                std::fs::rename(from, to)
            },
        )
        .await?,
    );

    // Populate our first store entry.
    let first_file_entry = {
        store.update_oneshot(digest, VALUE1.into()).await?;
        store.get_file_entry_for_digest(&digest).await?
    };

    // 1. Request the next rename function to block.
    // 2. Request to replace our data.
    // 3. When we are certain that our rename function is paused, drop
    //    the replace/update future.
    // 4. Then drop the lock.
    {
        let rename_pause_request_lock = RENAME_REQUEST_PAUSE_MUX.lock().await;
        let mut update_fut = store.update_oneshot(digest, VALUE2.into()).boxed();

        loop {
            // Try to advance our update future.
            assert_eq!(poll!(&mut update_fut), Poll::Pending);

            // Once we are sure the rename function is paused break.
            if RENAME_IS_PAUSED.load(Ordering::Acquire) {
                break;
            }
            // Give a little time for background/kernel threads to run.
            sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(
            store.get_part_unchunked(digest, 0, None).await?,
            VALUE1.as_bytes(),
            "the old generation must remain readable until the rename completes",
        );
        // Writing these out explicitly so users know this is what we are testing.
        // Note: The order they are dropped matters.
        drop(update_fut);
        drop(rename_pause_request_lock);
    }
    // Cancellation must not stop the background publish. The index changes
    // only once the rename has completed.
    let new_file_entry = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let entry = store.get_file_entry_for_digest(&digest).await?;
            if !Arc::ptr_eq(&first_file_entry, &entry) {
                return Ok::<_, Error>(entry);
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cancelled upload did not finish publishing")?;
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        VALUE2.as_bytes()
    );

    // Ensure the entry we inserted was properly flagged as moved (from temp -> content dir).
    new_file_entry
        .get_file_path_locked(move |file_path| async move {
            let expected_prefix = format!("{content_path}/{DIGEST_FOLDER_V2}/{digest}-");
            assert!(
                file_path.to_string_lossy().starts_with(&expected_prefix),
                "expected {file_path:?} to be a content file for {digest}"
            );
            Ok(())
        })
        .await?;

    Ok(())
}

#[nativelink_test]
async fn deleted_file_removed_from_store() -> Result<(), Error> {
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                read_buffer_size: 1,
                ..Default::default()
            },
            |from, to| std::fs::rename(from, to),
        )
        .await?,
    );

    store.update_oneshot(digest, VALUE1.into()).await?;

    let stored_file_path = content_file_path(&content_path, &StoreKey::Digest(digest)).await?;
    std::fs::remove_file(stored_file_path)?;

    let get_part_res = store.get_part_unchunked(digest, 0, None).await;
    assert_eq!(get_part_res.unwrap_err().code, Code::NotFound);

    // Repeat with a string typed key.

    let string_key = StoreKey::new_str(STRING_NAME);

    store
        .update_oneshot(string_key.borrow(), VALUE2.into())
        .await
        .unwrap();

    let stored_file_path =
        content_file_path(&content_path, &StoreKey::new_str(STRING_NAME)).await?;
    std::fs::remove_file(stored_file_path)?;

    let string_digest_get_part_res = store.get_part_unchunked(string_key, 0, None).await;
    assert_eq!(string_digest_get_part_res.unwrap_err().code, Code::NotFound);

    Ok(())
}

// Ensure that get_file_size() returns the correct number
// ceil(content length / block_size) * block_size
// assume block size 4K
// 1B data size = 4K size on disk
// 5K data size = 8K size on disk
#[nativelink_test]
async fn get_file_size_uses_block_size() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let value_1kb: String = "x".repeat(1024);
    let value_5kb: String = "xabcd".repeat(1024);

    let digest_1kb = DigestInfo::try_new(HASH1, value_1kb.len())?;
    let digest_5kb = DigestInfo::try_new(HASH2, value_5kb.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                read_buffer_size: 1,
                ..Default::default()
            },
            |from, to| std::fs::rename(from, to),
        )
        .await?,
    );

    store.update_oneshot(digest_1kb, value_1kb.into()).await?;
    let short_entry = store.get_file_entry_for_digest(&digest_1kb).await?;
    assert_eq!(short_entry.size_on_disk(), 4 * 1024);

    store.update_oneshot(digest_5kb, value_5kb.into()).await?;
    let long_entry = store.get_file_entry_for_digest(&digest_5kb).await?;
    assert_eq!(long_entry.size_on_disk(), 8 * 1024);
    Ok(())
}

#[nativelink_test]
async fn has_reports_content_length_while_eviction_counts_size_on_disk() -> Result<(), Error> {
    const BLOCK_SIZE: u64 = 4 * 1024;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let value_5kb: String = "xabcd".repeat(1024);
    let digest_5kb = DigestInfo::try_new(HASH2, value_5kb.len())?;
    store.update_oneshot(digest_5kb, value_5kb.into()).await?;
    store
        .update_oneshot(StoreKey::new_str("action_result"), "x".repeat(538).into())
        .await?;
    assert_eq!(
        store
            .get_file_entry_for_digest(&digest_5kb)
            .await?
            .size_on_disk(),
        2 * BLOCK_SIZE
    );
    assert_eq!(
        store.has(digest_5kb).await?,
        Some(5 * 1024),
        "has() must report the content length, not the size on disk"
    );
    assert_eq!(
        store.has(StoreKey::new_str("action_result")).await?,
        Some(538)
    );
    assert_eq!(store.has(StoreKey::new_str("missing")).await?, None);

    // Room for two one-block entries: the third pushes the total to max_bytes
    // and evicts the oldest. Counted by content length (1 + 2 + 3 bytes) all
    // three would stay.
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        eviction_policy: Some(EvictionPolicy {
            max_bytes: usize::try_from(3 * BLOCK_SIZE).unwrap(),
            ..Default::default()
        }),
        ..Default::default()
    })
    .await?;
    let digests = [1, 2, 3].map(|size| DigestInfo::try_new(HASH1, size).unwrap());
    for digest in digests {
        let data = make_random_data(usize::try_from(digest.size_bytes()).unwrap());
        store.update_oneshot(digest, data.into()).await?;
    }
    assert_eq!(
        store.has(digests[0]).await?,
        None,
        "each entry counts as a full block for eviction, so the oldest was evicted"
    );
    assert_eq!(store.has(digests[1]).await?, Some(2));
    assert_eq!(store.has(digests[2]).await?, Some(3));
    Ok(())
}

#[nativelink_test]
async fn update_with_whole_file_closes_file() -> Result<(), Error> {
    #[expect(clippy::collection_is_never_read)] // TODO(jhpratt) investigate
    let mut permits = vec![];
    // Grab all permits to ensure only 2 permits are available.
    {
        // We need two so the duplicate file work works!
        wait_for_no_open_files().await?;
        while fs::OPEN_FILE_SEMAPHORE.available_permits() > 2 {
            permits.push(fs::get_permit().await);
        }
        assert_eq!(
            fs::OPEN_FILE_SEMAPHORE.available_permits(),
            2,
            "Expected 2 permits to be available"
        );
    }
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let value = "x".repeat(1024);

    let digest = DigestInfo::try_new(HASH1, value.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            read_buffer_size: 1,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest, value.clone().into()).await?;

    let file_path = OsString::from(format!("{temp_path}/dummy_file"));
    let mut file = fs::create_file(&file_path).await?;
    {
        file.write_all(value.as_bytes()).await?;
        file.as_mut().sync_all().await?;
        file.seek(tokio::io::SeekFrom::Start(0)).await?;
    }

    store
        .update_with_whole_file(
            digest,
            file_path,
            file,
            UploadSizeInfo::ExactSize(value.len() as u64),
        )
        .await?;
    Ok(())
}

// Ensure that update_with_whole_file() moves the file without making a copy.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn update_with_whole_file_uses_same_inode() -> Result<(), Error> {
    use std::os::unix::fs::MetadataExt;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let value: String = "x".repeat(1024);

    let digest = DigestInfo::try_new(HASH1, value.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                read_buffer_size: 1,
                ..Default::default()
            },
            |from, to| std::fs::rename(from, to),
        )
        .await?,
    );

    let file_path = OsString::from(format!("{temp_path}/dummy_file"));
    let original_inode = {
        let mut file = fs::create_file(&file_path).await?;
        file.write_all(value.as_bytes()).await?;
        file.flush().await?;
        let original_inode = file.as_ref().metadata().await?.ino();

        let result = store
            .update_with_whole_file(
                digest,
                file_path,
                file,
                UploadSizeInfo::ExactSize(value.len() as u64),
            )
            .await?;
        assert!(
            result.1.is_none(),
            "Expected filesystem store to consume the file"
        );
        original_inode
    };

    let expected_file_name = content_file_path(&content_path, &StoreKey::Digest(digest)).await?;
    // Content blobs are stored read-only (0o444), so they cannot be opened for
    // write (`fs::create_file` opens read+write+truncate). Stat the path
    // directly to read the inode instead of opening the file.
    let new_inode = tokio::fs::metadata(&expected_file_name).await?.ino();
    assert_eq!(
        original_inode, new_inode,
        "Expected the same inode for the file"
    );

    Ok(())
}

#[nativelink_test]
async fn file_slot_taken_when_ready() -> Result<(), Error> {
    static FILE_SEMAPHORE: Semaphore = Semaphore::const_new(1);
    static WRITER_SEMAPHORE: Semaphore = Semaphore::const_new(1);
    static FILE_PERMIT: Mutex<Option<tokio::sync::SemaphorePermit<'_>>> = Mutex::new(None);
    static WRITER_PERMIT: Mutex<Option<tokio::sync::SemaphorePermit<'_>>> = Mutex::new(None);

    struct SingleSemaphoreHooks;
    impl FileEntryHooks for SingleSemaphoreHooks {
        async fn on_make_and_open(_encoded_file_path: &EncodedFilePath) -> Result<(), Error> {
            *FILE_PERMIT.lock() =
                Some(FILE_SEMAPHORE.acquire().await.map_err(|e| {
                    make_err!(Code::Internal, "Unable to acquire semaphore: {e:?}")
                })?);
            // Drop the writer permit now that we have one.
            WRITER_PERMIT.lock().take();
            Ok(())
        }
    }

    *WRITER_PERMIT.lock() = Some(WRITER_SEMAPHORE.acquire().await.unwrap());
    *FILE_PERMIT.lock() = Some(FILE_SEMAPHORE.acquire().await.unwrap());

    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let value_1: String = "x".repeat(1024);
    let value_2: String = "y".repeat(1024);

    let digest_1 = DigestInfo::try_new(HASH1, value_1.len())?;
    let digest_2 = DigestInfo::try_new(HASH2, value_2.len())?;

    let store = Box::pin(
        FilesystemStore::<TestFileEntry<SingleSemaphoreHooks>>::new_with_timeout_and_rename_fn(
            &FilesystemSpec {
                content_path: content_path.clone(),
                temp_path: temp_path.clone(),
                read_buffer_size: 1,
                ..Default::default()
            },
            |from, to| std::fs::rename(from, to),
        )
        .await?,
    );

    let value_1 = Bytes::from(value_1);
    let value_2 = Bytes::from(value_2);

    let (mut writer_1, reader_1) = make_buf_channel_pair();
    let (mut writer_2, reader_2) = make_buf_channel_pair();
    let size_1 = UploadSizeInfo::ExactSize(value_1.len().try_into()?);
    let size_2 = UploadSizeInfo::ExactSize(value_2.len().try_into()?);
    let store_ref = &store;
    let update_1_fut = async move {
        let result = store_ref.update(digest_1, reader_1, size_1).await;
        FILE_PERMIT.lock().take();
        result
    };
    let update_2_fut = async move {
        let result = store_ref.update(digest_2, reader_2, size_2).await;
        FILE_PERMIT.lock().take();
        result
    };

    let writer_1_fut = async move {
        let _permit = WRITER_SEMAPHORE.acquire().await.unwrap();
        writer_1.send(value_1.slice(0..1)).await?;
        writer_1.send(value_1.slice(1..2)).await?;
        writer_1.send(value_1.slice(2..3)).await?;
        writer_1.send(value_1.slice(3..)).await?;
        writer_1.send_eof()?;
        Ok::<_, Error>(())
    };
    let writer_2_fut = async move {
        writer_2.send(value_2.slice(0..1)).await?;
        writer_2.send(value_2.slice(1..2)).await?;
        writer_2.send(value_2.slice(2..3)).await?;
        // Allow the update to get a file permit.
        FILE_PERMIT.lock().take();
        writer_2.send(value_2.slice(3..)).await?;
        writer_2.send_eof()?;
        Ok::<_, Error>(())
    };

    let (res_1, res_2, res_3, res_4) = tokio::time::timeout(Duration::from_secs(10), async move {
        tokio::join!(update_1_fut, update_2_fut, writer_1_fut, writer_2_fut)
    })
    .await
    .map_err(|err| Error::from_std_err(Code::Internal, &err).append("Deadlock detected"))?;
    res_1.merge(res_2).merge(res_3).merge(res_4)
}

// If we insert a file larger than the max_bytes eviction policy, it should be safely
// evicted, without deadlocking.
#[nativelink_test]
async fn safe_small_safe_eviction() -> Result<(), Error> {
    let store_spec = FilesystemSpec {
        content_path: "/tmp/nativelink/safe_fs".into(),
        temp_path: "/tmp/nativelink/safe_fs_temp".into(),
        eviction_policy: Some(EvictionPolicy {
            max_bytes: 1,
            ..Default::default()
        }),
        ..Default::default()
    };
    let store = Store::new(<FilesystemStore>::new(&store_spec).await?);

    // > than the max_bytes
    let bytes = 2;

    let data = make_random_data(bytes);
    let digest = DigestInfo::try_new(VALID_HASH, data.len()).unwrap();

    assert_eq!(
        store.has(digest).await,
        Ok(None),
        "Expected data to not exist in store"
    );

    store.update_oneshot(digest, data.clone().into()).await?;

    assert_eq!(
        store.has(digest).await,
        Ok(None),
        "Expected data to not exist in store, because eviction"
    );

    let (tx, mut rx) = make_buf_channel_pair();

    assert_eq!(
        store.get(digest, tx).await,
        Err(Error {
            code: Code::NotFound,
            messages: vec![format!(
                "{VALID_HASH}-{bytes} not found in filesystem store here"
            )],
            context: ErrorContext::None,
        }),
        "Expected data to not exist in store, because eviction"
    );

    assert!(rx.recv().await.is_err());

    Ok(())
}

#[nativelink_test]
async fn add_too_early_files() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let demo_file_folder = format!("{content_path}/{STR_FOLDER_V2}");
    fs::create_dir_all(&demo_file_folder).await?;
    let demo_file_path = format!("{demo_file_folder}/foo-0");
    std::fs::write(&demo_file_path, "demo text")
        .err_tip(|| format!("writing to {demo_file_path}"))?;
    debug!(%demo_file_path, "demo file path");

    // Add 60 seconds to the access time to trigger the logging message about access times
    fs_set_times::set_atime(
        &demo_file_path,
        SystemTime::now()
            .checked_add(Duration::from_mins(1))
            .unwrap()
            .into(),
    )?;

    FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        read_buffer_size: 1,
        ..Default::default()
    })
    .await
    .err_tip(|| "during FileSystemStore::new")?;

    assert!(logs_contain(
        "File access time newer than FilesystemStore start time file_name=foo-0 atime=20"
    ));

    Ok(())
}

#[nativelink_test]
async fn check_health_ok_when_content_path_is_a_directory() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    fs::create_dir_all(&content_path).await?;
    fs::create_dir_all(&temp_path).await?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path,
        eviction_policy: None,
        ..Default::default()
    })
    .await?;

    match HealthStatusIndicator::check_health(&*store, Cow::Borrowed("test")).await {
        HealthStatus::Ok { .. } => Ok(()),
        other => panic!("expected HealthStatus::Ok, got {other:?}"),
    }
}

#[nativelink_test]
async fn check_health_failed_when_content_path_is_missing() -> Result<(), Error> {
    // Construct the store against a real path, then delete it so the
    // stat() inside check_health fails with ENOENT.
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    fs::create_dir_all(&content_path).await?;
    fs::create_dir_all(&temp_path).await?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path,
        eviction_policy: None,
        ..Default::default()
    })
    .await?;

    fs::remove_dir_all(&content_path).await?;

    match HealthStatusIndicator::check_health(&*store, Cow::Borrowed("test")).await {
        HealthStatus::Failed { message, .. } => {
            assert!(
                message.contains("stat") || message.contains("not a directory"),
                "unexpected failure message: {message}",
            );
            Ok(())
        }
        other => panic!("expected HealthStatus::Failed, got {other:?}"),
    }
}

/// `get_executable_hardlink_source` must return a private **0o555** inode that
/// is created exactly once: distinct from the read-only **0o444** CAS blob,
/// stable across calls (so callers hardlink-many), and leaving the CAS blob's
/// mode/inode untouched. This is what lets the worker materialize executable
/// inputs by hardlink alone — with no per-action writable fd — which is the
/// fix for the `ETXTBSY` ("Text file busy") race on executable inputs.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn executable_hardlink_source_created_once_and_readonly() -> Result<(), Error> {
    use std::os::unix::fs::MetadataExt;

    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest, VALUE1.into()).await?;

    // The CAS blob itself is stored read-only 0o444.
    let blob_path = content_file_path(&content_path, &StoreKey::Digest(digest)).await?;
    let blob_meta = fs::metadata(&blob_path).await?;
    assert_eq!(
        blob_meta.mode() & 0o777,
        0o444,
        "CAS blob must be stored read-only 0o444"
    );
    let blob_ino = blob_meta.ino();

    // First call materializes the executable variant: a separate 0o555 inode
    // with identical content.
    let exec1 = store.get_executable_hardlink_source(&digest).await?;
    let exec_meta1 = fs::metadata(&exec1).await?;
    assert_eq!(
        exec_meta1.mode() & 0o777,
        0o555,
        "executable variant must be 0o555 (read + execute, no write)"
    );
    assert_ne!(
        exec_meta1.ino(),
        blob_ino,
        "executable variant must be a private inode, not the shared CAS blob"
    );
    assert_eq!(
        read_file_contents(exec1.as_os_str()).await?,
        VALUE1.as_bytes(),
        "executable variant content must match the CAS blob"
    );

    // Second call is created-once: same path and same inode (callers hardlink
    // this one inode many times rather than re-copying per action).
    let exec2 = store.get_executable_hardlink_source(&digest).await?;
    assert_eq!(exec1, exec2, "executable variant path must be stable");
    assert_eq!(
        fs::metadata(&exec2).await?.ino(),
        exec_meta1.ino(),
        "second call must reuse the same variant inode (created once)"
    );

    // Creating the variant must not have mutated the shared CAS blob.
    let blob_meta_after = fs::metadata(&blob_path).await?;
    assert_eq!(
        blob_meta_after.mode() & 0o777,
        0o444,
        "CAS blob mode must be untouched by variant creation"
    );
    assert_eq!(
        blob_meta_after.ino(),
        blob_ino,
        "CAS blob inode must be untouched by variant creation"
    );

    // Hardlinking the variant yields an executable that shares its inode — the
    // hardlink-only materialization the worker now performs.
    let dest = OsString::from(format!("{temp_path}/materialized_exec"));
    fs::hard_link(&exec1, &dest).await?;
    let dest_meta = fs::metadata(&dest).await?;
    assert_eq!(
        dest_meta.ino(),
        exec_meta1.ino(),
        "hardlinked executable must share the variant inode"
    );
    assert_eq!(
        dest_meta.mode() & 0o111,
        0o111,
        "materialized executable must carry the +x bits"
    );

    Ok(())
}

/// Which renames a [`RenameGate`] scripts. Every other rename passes straight
/// through.
#[cfg(target_family = "unix")]
#[derive(Clone, Copy, Debug)]
enum GatedRenames {
    /// Renames publishing executable variants into `{content_path}.exec`.
    ExecutableVariants,
    /// Renames publishing ordinary blobs into the content directory.
    ContentPublishes,
}

#[cfg(target_family = "unix")]
impl GatedRenames {
    fn selects(self, to: &OsStr) -> bool {
        let into_executable_variants = to.to_string_lossy().contains(".exec/");
        match self {
            Self::ExecutableVariants => into_executable_variants,
            Self::ContentPublishes => !into_executable_variants,
        }
    }
}

/// Scripted stand-in for `rename(2)` that lets a test hold or fail one kind of
/// rename ([`GatedRenames`]), in order. Each test owns its own `static` gate,
/// because a store takes its rename hook as a plain `fn` pointer.
///
/// Only the writers a test expects can be held. A rename beyond that count is
/// an extra writer: it fails at once instead of waiting, so a defect that adds
/// a writer — however late it arrives — ends the test with a failure rather
/// than leaving it awaiting a writer that no release will ever reach.
#[cfg(target_family = "unix")]
#[derive(Debug)]
struct RenameGate {
    gated: GatedRenames,
    state: std::sync::Mutex<RenameGateState>,
    changed: std::sync::Condvar,
}

#[cfg(target_family = "unix")]
#[derive(Debug)]
struct RenameGateState {
    /// Gated renames that have started.
    entered: u32,
    /// Gated rename `i` proceeds only once `i < released`.
    released: u32,
    /// Bit `i` makes gated rename `i` fail instead of renaming.
    fail_mask: u32,
    /// Gated renames the test expects; rename `i >= expected` is an
    /// extra writer and fails immediately without waiting.
    expected: u32,
}

#[cfg(target_family = "unix")]
impl RenameGate {
    const fn new(gated: GatedRenames, released: u32, fail_mask: u32, expected: u32) -> Self {
        Self {
            gated,
            state: std::sync::Mutex::new(RenameGateState {
                entered: 0,
                released,
                fail_mask,
                expected,
            }),
            changed: std::sync::Condvar::new(),
        }
    }

    fn rename(&self, from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        if !self.gated.selects(to) {
            return std::fs::rename(from, to);
        }
        let mut state = self.state.lock().unwrap();
        let index = state.entered;
        state.entered += 1;
        self.changed.notify_all();
        if index >= state.expected {
            return Err(std::io::Error::other(format!(
                "unexpected extra writer (rename {index})"
            )));
        }
        while index >= state.released {
            state = self.changed.wait(state).unwrap();
        }
        let fail = state.fail_mask & (1 << index) != 0;
        drop(state);
        if fail {
            return Err(std::io::Error::other("publish blocked by test"));
        }
        std::fs::rename(from, to)
    }

    fn entered(&self) -> u32 {
        self.state.lock().unwrap().entered
    }

    fn released(&self) -> u32 {
        self.state.lock().unwrap().released
    }

    fn release(&self, released: u32) {
        self.state.lock().unwrap().released = released;
        self.changed.notify_all();
    }

    /// Waits, without blocking the test runtime, until `count` gated
    /// renames have started.
    async fn wait_entered(&self, count: u32) -> Result<(), Error> {
        for _ in 0..10_000 {
            if self.entered() >= count {
                return Ok(());
            }
            sleep(Duration::from_millis(1)).await;
        }
        Err(make_err!(
            Code::DeadlineExceeded,
            "Only {} of {count} gated renames started",
            self.entered()
        ))
    }

    /// Blocks until the test has released `released` renames, or until
    /// `limit` passes, then releases them itself. Returns whether it had to.
    /// Run it on a blocking-pool thread: a test whose runtime thread is stuck
    /// inside a held rename cannot release it, so this is what ends that test.
    fn release_after(&self, limit: Duration, released: u32) -> bool {
        let state = self.state.lock().unwrap();
        let (mut state, wait) = self
            .changed
            .wait_timeout_while(state, limit, |state| state.released < released)
            .unwrap();
        if wait.timed_out() {
            state.released = released;
            self.changed.notify_all();
        }
        wait.timed_out()
    }
}

/// Releases every held rename when dropped, so a failing assertion cannot
/// strand a blocking-pool thread (and hang runtime shutdown) inside the gate.
#[cfg(target_family = "unix")]
#[derive(Debug)]
struct ReleaseAllOnDrop(&'static RenameGate);

#[cfg(target_family = "unix")]
impl Drop for ReleaseAllOnDrop {
    fn drop(&mut self) {
        self.0.release(u32::MAX);
    }
}

/// How long a test waits for a task it owns before failing instead.
#[cfg(target_family = "unix")]
const TASK_JOIN_LIMIT: Duration = Duration::from_mins(1);

/// Joins a test-owned task, failing instead of waiting forever if it never
/// finishes. The gate already rules out a hang inside it; this bounds the
/// test against any other.
#[cfg(target_family = "unix")]
async fn join_bounded<T>(task: tokio::task::JoinHandle<T>) -> T {
    tokio::time::timeout(TASK_JOIN_LIMIT, task)
        .await
        .expect("test-owned task did not finish within TASK_JOIN_LIMIT")
        .expect("test-owned task panicked")
}

/// Names in the executable-variant directory: published variants and any
/// temporary file a materialization left behind.
#[cfg(target_family = "unix")]
fn executable_dir_entries(content_path: &str) -> Result<Vec<String>, Error> {
    let mut names = std::fs::read_dir(format!("{content_path}.exec/{DIGEST_FOLDER_V2}"))?
        .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    names.sort();
    Ok(names)
}

/// The variant this process may hardlink must have no open writer anywhere in
/// the process: an inherited writable fd is what turns a later `execve` into
/// `ETXTBSY`.
#[cfg(target_os = "linux")]
fn assert_no_open_fd_for(path: &OsStr) -> Result<(), Error> {
    let target = std::fs::canonicalize(path)?;
    for entry in std::fs::read_dir("/proc/self/fd")? {
        if let Ok(link) = std::fs::read_link(entry?.path()) {
            assert_ne!(link, target, "published executable variant has an open fd");
        }
    }
    Ok(())
}

/// A failed attempt to publish an executable variant must not poison its
/// digest. The attempt copies the read-only (0o444) CAS blob to a temporary
/// file, makes it 0o555 and renames it into place; when the rename fails, the
/// next attempt for the same digest must still build and publish the variant,
/// and nothing from the failed attempt may remain. Before the fix the failed
/// attempt left its fixed `.tmp` path behind read-only, so every retry's copy
/// into it failed with `EACCES` for as long as that leftover persisted.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn executable_variant_retry_succeeds_after_failed_publish() -> Result<(), Error> {
    use std::os::unix::fs::MetadataExt;

    static GATE: RenameGate = RenameGate::new(GatedRenames::ExecutableVariants, u32::MAX, 0b1, 2);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    const SCRIPT: &str = "#!/bin/sh\necho executable-variant-ran\n";
    let _release = ReleaseAllOnDrop(&GATE);

    let content_path = make_temp_path("content_path");
    let digest = DigestInfo::try_new(HASH1, SCRIPT.len())?;
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    store.update_oneshot(digest, SCRIPT.into()).await?;
    let blob_path = content_file_path(&content_path, &StoreKey::Digest(digest)).await?;
    let blob_meta = fs::metadata(&blob_path).await?;
    assert_eq!(blob_meta.mode() & 0o777, 0o444);

    // The worker resolves executable inputs through the batch entry point.
    let mut first = store.get_executable_hardlink_sources(&[digest]).await;
    let first_err = first.pop().unwrap().expect_err("injected rename failure");
    assert!(
        format!("{first_err:?}").contains("executable-variant rename failed"),
        "unexpected first error: {first_err:?}"
    );
    assert_eq!(GATE.entered(), 1);

    let retry = store
        .get_executable_hardlink_sources(&[digest])
        .await
        .pop()
        .unwrap()
        .err_tip(|| "Retrying the executable variant after a failed publish")?;
    assert_eq!(GATE.entered(), 2, "the retry must publish exactly once");

    let variant_meta = fs::metadata(&retry).await?;
    assert_eq!(variant_meta.mode() & 0o777, 0o555);
    assert_ne!(variant_meta.ino(), blob_meta.ino());
    assert_eq!(read_file_contents(&retry).await?, SCRIPT.as_bytes());
    assert_eq!(
        executable_dir_entries(&content_path)?,
        vec![digest.to_string()],
        "only the published variant may remain; temporary files must be removed"
    );
    #[cfg(target_os = "linux")]
    assert_no_open_fd_for(&retry)?;
    // The published variant runs: no writer holds it (`ETXTBSY`) and it is
    // executable (`EACCES`).
    let ran = std::process::Command::new(&retry).output()?;
    assert!(ran.status.success(), "variant failed to run: {ran:?}");
    assert_eq!(ran.stdout, b"executable-variant-ran\n");

    // Warm reuse: the same inode, with no further materialization.
    let warm = store.get_executable_hardlink_source(&digest).await?;
    assert_eq!(warm, retry);
    assert_eq!(fs::metadata(&warm).await?.ino(), variant_meta.ino());
    assert_eq!(GATE.entered(), 2);

    // The shared CAS blob is untouched throughout.
    let blob_meta_after = fs::metadata(&blob_path).await?;
    assert_eq!(blob_meta_after.mode() & 0o777, 0o444);
    assert_eq!(blob_meta_after.ino(), blob_meta.ino());
    assert_eq!(read_file_contents(&blob_path).await?, SCRIPT.as_bytes());
    Ok(())
}

/// Cancelling a caller must not let a second writer start while the first
/// materialization is still running. The copy runs on a blocking thread that
/// outlives a cancelled caller, so that thread, not the caller, must keep the
/// digest's single-flight lock until it has finished. A later caller then
/// waits and reuses the published variant rather than copying again.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn cancelled_executable_variant_caller_keeps_single_writer() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new(GatedRenames::ExecutableVariants, 0, 0, 1);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _release = ReleaseAllOnDrop(&GATE);

    let content_path = make_temp_path("content_path");
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    store.update_oneshot(digest, VALUE1.into()).await?;

    // The first caller copies and chmods, then holds in the rename; cancel it
    // there. Awaiting the aborted task proves its future has been dropped.
    let first = background_spawn!("cancelled_variant_first", {
        let store = store.clone();
        async move { store.get_executable_hardlink_source(&digest).await }
    });
    GATE.wait_entered(1).await?;
    first.abort();
    let cancelled = tokio::time::timeout(TASK_JOIN_LIMIT, first)
        .await
        .expect("cancelled task did not finish within TASK_JOIN_LIMIT");
    assert!(cancelled.is_err_and(|e| e.is_cancelled()));

    // While the cancelled caller's copy is unfinished, a new caller must wait
    // for it instead of starting a second writer.
    let mut second = background_spawn!("cancelled_variant_second", {
        let store = store.clone();
        async move { store.get_executable_hardlink_source(&digest).await }
    });
    if let Ok(early) = tokio::time::timeout(Duration::from_secs(2), &mut second).await {
        panic!("second caller finished while the first writer was unfinished: {early:?}");
    }
    // Witness that the waiting caller did not write either: a second writer
    // would by now have copied and reached its own (held) rename.
    assert_eq!(
        GATE.entered(),
        1,
        "a second writer started while the first was unfinished"
    );
    GATE.release(1);
    let variant = join_bounded(second).await?;
    assert_eq!(
        GATE.entered(),
        1,
        "exactly one writer may materialize the variant"
    );
    assert_eq!(read_file_contents(&variant).await?, VALUE1.as_bytes());
    assert_eq!(
        executable_dir_entries(&content_path)?,
        vec![digest.to_string()]
    );
    Ok(())
}

/// After a failed materialization, callers that were already waiting on the
/// failed attempt's lock and callers arriving afresh must still serialize on
/// one lock. Before the fix the failed attempt removed the lock entry while
/// old waiters held it, so a newcomer created a second lock and wrote
/// alongside an old waiter.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn executable_variant_waiters_and_newcomers_share_one_writer() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new(GatedRenames::ExecutableVariants, 0, 0b1, 2);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _release = ReleaseAllOnDrop(&GATE);

    let content_path = make_temp_path("content_path");
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    let spawn_lookup = || {
        let store = store.clone();
        background_spawn!("variant_lookup", async move {
            store.get_executable_hardlink_source(&digest).await
        })
    };

    // The first writer holds in its rename while two more callers queue on
    // its lock; then its rename fails.
    let failing = spawn_lookup();
    GATE.wait_entered(1).await?;
    let waiters = [spawn_lookup(), spawn_lookup()];
    // Give both callers time to pass the fast-path check and queue on the
    // writer's lock. Nothing observable marks that, so this settles rather
    // than proves it; the test passes whichever way they arrive, and a
    // mutation run that restores the old lock removal shows it discriminates.
    sleep(Duration::from_millis(500)).await;
    GATE.release(1);
    let failed = join_bounded(failing)
        .await
        .expect_err("injected rename failure");
    assert!(format!("{failed:?}").contains("executable-variant rename failed"));

    // One old waiter takes over and holds in its own rename. A newcomer must
    // queue behind it rather than start a second writer.
    GATE.wait_entered(2).await?;
    let mut newcomer = spawn_lookup();
    if let Ok(early) = tokio::time::timeout(Duration::from_secs(2), &mut newcomer).await {
        panic!("newcomer finished while an old waiter was writing: {early:?}");
    }
    // Witness that nobody else wrote alongside the old waiter: another writer
    // would by now have reached its own (held) rename.
    assert_eq!(
        GATE.entered(),
        2,
        "a second writer started alongside the old waiter"
    );
    GATE.release(2);

    let variant = join_bounded(newcomer).await?;
    for waiter in waiters {
        assert_eq!(join_bounded(waiter).await?, variant);
    }
    assert_eq!(
        GATE.entered(),
        2,
        "one failed and one successful writer only"
    );
    assert_eq!(read_file_contents(&variant).await?, VALUE1.as_bytes());
    assert_eq!(
        executable_dir_entries(&content_path)?,
        vec![digest.to_string()]
    );
    Ok(())
}

/// A store whose `.exec` directory cannot be cleared must still start and
/// serve ordinary CAS reads. It must not, however, trust an executable variant
/// left behind by an earlier process: that file may be stale or torn after an
/// unclean shutdown, so the executable hardlink fast path stays disabled until
/// the directory can be wiped on a later writable startup.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn read_only_executable_dir_preserves_cas_reads_without_reusing_variants() -> Result<(), Error>
{
    use std::os::unix::fs::PermissionsExt;

    struct RestoreWritable(String);

    impl Drop for RestoreWritable {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755))
                .expect("restore executable digest directory permissions");
        }
    }

    drop_all_fs_read_caps()?;

    let content_path = make_temp_path("content_path_read_only_exec");
    let temp_path = make_temp_path("temp_path_read_only_exec");
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let spec = FilesystemSpec {
        content_path: content_path.clone(),
        temp_path,
        ..Default::default()
    };

    {
        let store = FilesystemStore::<FileEntryImpl>::new(&spec).await?;
        store.update_oneshot(digest, VALUE1.into()).await?;
        store.get_executable_hardlink_source(&digest).await?;
    }

    let executable_digest_dir = format!("{content_path}.exec/{DIGEST_FOLDER_V2}");
    fs::set_permissions(
        &executable_digest_dir,
        std::fs::Permissions::from_mode(0o555),
    )
    .await?;
    let _restore_permissions = RestoreWritable(executable_digest_dir);

    let store = FilesystemStore::<FileEntryImpl>::new(&spec).await?;
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        VALUE1.as_bytes(),
        "read-only executable cleanup must not prevent ordinary CAS reads"
    );
    debug!("Reading executable hardlink source for {digest:?}");
    let res = store.get_executable_hardlink_source(&digest).await;
    debug!("Hardlink result is {res:?}");
    let err = res.expect_err("a surviving executable variant must not be trusted");
    assert_eq!(err.code, Code::FailedPrecondition);

    Ok(())
}

/// Regression test for #2474: the `.exec` variant directory was never
/// registered in `evicting_map`, so it was invisible to `max_bytes` and only
/// ever cleared by the startup `remove_dir_all` — growing without bound at
/// runtime. Evicting a digest from the primary CAS must also delete its
/// `.exec` sibling, bounding total `.exec` disk use to the primary store's
/// own eviction policy instead of letting it grow forever.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn evicting_digest_deletes_its_executable_variant() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let digest2 = DigestInfo::try_new(HASH2, VALUE2.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: Some(EvictionPolicy {
                max_count: 1,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest1, VALUE1.into()).await?;

    let variant_path = OsString::from(format!("{content_path}.exec/{DIGEST_FOLDER_V2}/{digest1}"));
    store.get_executable_hardlink_source(&digest1).await?;
    fs::metadata(&variant_path)
        .await
        .err_tip(|| "Executable variant should exist right after creation")?;

    // max_count: 1 means inserting a second digest evicts the first from the
    // primary evicting_map, which must also delete digest1's `.exec` sibling.
    store.update_oneshot(digest2, VALUE2.into()).await?;

    let err = fs::metadata(&variant_path)
        .await
        .expect_err("Executable variant must be deleted when its digest is evicted");
    assert_eq!(
        err.code,
        Code::NotFound,
        "Expected the executable variant to be gone, got: {err:?}"
    );

    Ok(())
}

/// The warm variant path never consults the eviction index, so a toolchain
/// binary that every action hardlinks from its variant looked idle to the LRU
/// and was evicted first when the store filled, taking its variant with it
/// mid-batch. A variant hit must count as a use of the blob.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn executable_variant_hit_refreshes_blob_recency() -> Result<(), Error> {
    const HASH3: &str = "0123456789abcdef000000000000000000030000000000000123456789abcdef";
    const VALUE3: &str = "5555555555";
    let content_path = make_temp_path("content_path");
    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let digest2 = DigestInfo::try_new(HASH2, VALUE2.len())?;
    let digest3 = DigestInfo::try_new(HASH3, VALUE3.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: make_temp_path("temp_path"),
            eviction_policy: Some(EvictionPolicy {
                max_count: 2,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest1, VALUE1.into()).await?;
    store.get_executable_hardlink_source(&digest1).await?;
    store.update_oneshot(digest2, VALUE2.into()).await?;

    // digest1 is the older entry, but its variant is used again here.
    let sources = store.get_executable_hardlink_sources(&[digest1]).await;
    assert_eq!(sources.len(), 1);
    let variant_path = sources.into_iter().next().unwrap()?;
    assert_eq!(
        variant_path,
        OsString::from(format!("{content_path}.exec/{DIGEST_FOLDER_V2}/{digest1}"))
    );

    // The third insert evicts one entry; it must be the one not used since.
    store.update_oneshot(digest3, VALUE3.into()).await?;
    assert!(
        store.has(digest1).await?.is_some(),
        "the blob behind a used variant must stay resident"
    );
    assert!(
        store.has(digest2).await?.is_none(),
        "the entry not used since must be the one evicted"
    );
    fs::metadata(&variant_path)
        .await
        .err_tip(|| "the used variant must survive the eviction")?;
    Ok(())
}

/// A variant left on disk after its blob dropped out of the index is not a
/// hit: handing it out would keep an untracked file in use for good. It is a
/// miss, so the caller fills the fast store again and the blob is indexed.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn orphaned_executable_variant_is_a_miss() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        })
        .await?,
    );
    // The variant without its blob, as a deletion that never ran leaves it.
    let variant_path = format!("{content_path}.exec/{DIGEST_FOLDER_V2}/{digest}");
    fs::create_dir_all(format!("{content_path}.exec/{DIGEST_FOLDER_V2}")).await?;
    std::fs::write(&variant_path, VALUE1)?;

    let sources = store.get_executable_hardlink_sources(&[digest]).await;
    assert_eq!(sources.len(), 1);
    let err = sources
        .into_iter()
        .next()
        .unwrap()
        .expect_err("an orphaned variant must not be handed out");
    assert_eq!(err.code, Code::NotFound, "{err:?}");
    Ok(())
}

/// Waits until `store`'s map has no entries: a removal has taken its entry
/// out, and its callbacks may still be waiting.
#[cfg(target_family = "unix")]
async fn wait_until_map_empty(store: &FilesystemStore<FileEntryImpl>) -> Result<(), Error> {
    let evicting_map = store.get_evicting_map();
    tokio::time::timeout(TASK_JOIN_LIMIT, async {
        while evicting_map.len_for_test() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .map_err(|_| {
        make_err!(
            Code::DeadlineExceeded,
            "the removal never took its entry out"
        )
    })
}

/// Starts removing `digest`'s only generation while an earlier retirement
/// holds the store's retirement admission, and returns once the entry is out
/// of the map: its executable-variant cleanup is now waiting for admission.
#[cfg(target_family = "unix")]
async fn start_delayed_removal(
    store: &Arc<FilesystemStore<FileEntryImpl>>,
    digest: DigestInfo,
) -> Result<
    (
        tokio::sync::OwnedSemaphorePermit,
        tokio::task::JoinHandle<bool>,
    ),
    Error,
> {
    let admission = store.hold_retirement_admission_for_test().await;
    let removal = background_spawn!("delayed_removal", {
        let evicting_map = store.get_evicting_map();
        async move { evicting_map.remove(&StoreKey::Digest(digest)).await }
    });
    wait_until_map_empty(store).await?;
    assert!(
        !removal.is_finished(),
        "the removal's cleanup must be waiting"
    );
    Ok((admission, removal))
}

/// Hardlinks `variant` the way a worker materializes an executable input, and
/// checks the link's bytes and mode.
#[cfg(target_family = "unix")]
async fn hardlink_executable_input(variant: &OsStr, input: &str) -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::hard_link(variant, input).err_tip(|| "Hardlinking the executable variant")?;
    assert_eq!(
        read_file_contents(OsStr::new(input)).await?,
        VALUE1.as_bytes()
    );
    let mode = std::fs::metadata(input)?.permissions().mode();
    assert_eq!(mode & 0o777, 0o555, "executable input must be 0o555");
    Ok(())
}

/// A removal's variant cleanup can wait for retirement admission long after
/// its entry left the map. Meanwhile a worker may lease the digest, upload it
/// again and resolve its executable variant (creating one, or finding one the
/// old generation left). The old cleanup must not delete the variant the
/// resident, leased generation's worker is about to hardlink.
#[cfg(target_family = "unix")]
async fn check_delayed_old_removal_keeps_new_generations_variant(
    old_generation_had_variant: bool,
) -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    if old_generation_had_variant {
        store.get_executable_hardlink_source(&digest).await?;
    }
    let (admission, removal) = start_delayed_removal(&store, digest).await?;

    store.lease_digest(&digest);
    store.update_oneshot(digest, VALUE1.into()).await?;
    let resolve = background_spawn!("resolve_new_generations_variant", {
        let store = store.clone();
        async move { store.get_executable_hardlink_source(&digest).await }
    });
    // Admit the old cleanup only once the worker has resolved its variant, or
    // is queued for the variant lease behind someone holding it (the old
    // cleanup is the only other claimant).
    tokio::time::timeout(TASK_JOIN_LIMIT, async {
        while !resolve.is_finished() && store.executable_variant_lease_claims_for_test(&digest) < 2
        {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the worker neither resolved its variant nor queued for it");
    drop(admission);
    assert!(join_bounded(removal).await);
    let variant = join_bounded(resolve).await?;

    assert!(
        store
            .get_evicting_map()
            .is_resident(&StoreKey::Digest(digest)),
        "the new, leased generation stays resident"
    );
    hardlink_executable_input(&variant, &format!("{content_path}-input")).await?;
    store.release_digests(&[digest]).await;
    Ok(())
}

#[cfg(target_family = "unix")]
#[nativelink_test]
async fn delayed_old_removal_keeps_variant_created_for_new_generation() -> Result<(), Error> {
    check_delayed_old_removal_keeps_new_generations_variant(false).await
}

#[cfg(target_family = "unix")]
#[nativelink_test]
async fn delayed_old_removal_keeps_variant_reused_by_new_generation() -> Result<(), Error> {
    check_delayed_old_removal_keeps_new_generations_variant(true).await
}

/// A variant whose digest is not resident again when its delayed cleanup is
/// admitted is still deleted.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn delayed_removal_deletes_variant_of_digest_not_resident() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    let variant = store.get_executable_hardlink_source(&digest).await?;
    let (admission, removal) = start_delayed_removal(&store, digest).await?;
    drop(admission);
    assert!(join_bounded(removal).await);
    assert!(
        !Path::new(&variant).exists(),
        "the variant of a digest no longer resident must be deleted"
    );
    assert_eq!(executable_dir_entries(&content_path)?, Vec::<String>::new());
    Ok(())
}

/// Cancelling a removal while its variant cleanup waits for admission deletes
/// nothing and leaves the digest's variant lease free: a later generation
/// resolves the variant at once, and that generation's removal deletes it.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn cancelled_delayed_removal_leaves_variant_lease_free() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    let variant = store.get_executable_hardlink_source(&digest).await?;
    let (admission, removal) = start_delayed_removal(&store, digest).await?;
    removal.abort();
    let cancelled = tokio::time::timeout(TASK_JOIN_LIMIT, removal)
        .await
        .expect("cancelled removal did not finish within TASK_JOIN_LIMIT");
    assert!(cancelled.is_err_and(|e| e.is_cancelled()));
    drop(admission);

    store.update_oneshot(digest, VALUE1.into()).await?;
    let resolved = tokio::time::timeout(
        Duration::from_secs(2),
        store.get_executable_hardlink_source(&digest),
    )
    .await
    .expect("the cancelled cleanup must not leave the variant lease held")?;
    assert_eq!(resolved, variant);
    hardlink_executable_input(&resolved, &format!("{content_path}-input")).await?;

    assert!(
        store
            .get_evicting_map()
            .remove(&StoreKey::Digest(digest))
            .await
    );
    assert!(!Path::new(&variant).exists());
    Ok(())
}

/// A remove callback that holds the cleanup of every entry removed after
/// `key` in the same removal: removals run their entries' callbacks one at a
/// time, so while this waits, a later entry's executable-variant cleanup has
/// not started and holds no variant lease.
#[cfg(target_family = "unix")]
#[derive(Debug)]
struct CleanupGate {
    key: DigestInfo,
    entered: Arc<Semaphore>,
    release: Arc<Semaphore>,
}

#[cfg(target_family = "unix")]
impl RemoveItemCallback for CleanupGate {
    fn callback<'a>(
        &'a self,
        store_key: StoreKey<'a>,
    ) -> core::pin::Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            if store_key != StoreKey::Digest(self.key) {
                return;
            }
            self.entered.add_permits(1);
            drop(self.release.acquire().await);
        })
    }
}

/// Which source a worker's executable lookup goes through.
#[cfg(target_family = "unix")]
#[derive(Clone, Copy, Debug)]
enum ExecutableLookup {
    /// [`FilesystemStore::get_executable_hardlink_sources`], the worker's
    /// batched input-tree path.
    Batch,
    /// [`FilesystemStore::get_executable_hardlink_source`].
    Single,
}

#[cfg(target_family = "unix")]
async fn lookup_executable(
    store: &FilesystemStore<FileEntryImpl>,
    lookup: ExecutableLookup,
    digest: DigestInfo,
) -> Result<OsString, Error> {
    match lookup {
        ExecutableLookup::Batch => store
            .get_executable_hardlink_sources(&[digest])
            .await
            .pop()
            .expect("one result per digest"),
        ExecutableLookup::Single => store.get_executable_hardlink_source(&digest).await,
    }
}

/// Resolves `digest`'s executable source the way the worker's
/// `resolve_exec_files` does: a failed lookup means the blob is not resident,
/// so the worker populates the fast store (here, uploads it again) and
/// resolves once more. Returns the source and whether the first lookup
/// already accepted it.
#[cfg(target_family = "unix")]
async fn resolve_executable_like_worker(
    store: &FilesystemStore<FileEntryImpl>,
    lookup: ExecutableLookup,
    digest: DigestInfo,
    value: &'static str,
) -> Result<(OsString, bool), Error> {
    if let Ok(source) = lookup_executable(store, lookup, digest).await {
        return Ok((source, true));
    }
    store.update_oneshot(digest, value.into()).await?;
    let source = store
        .get_executable_hardlink_source(&digest)
        .await
        .err_tip(|| "Resolving executable variant after populating")?;
    Ok((source, false))
}

/// Where the digest's primary entry stands when the worker resolves its
/// executable source, after the digest's old removal has taken the entry out
/// but before that removal's variant cleanup has started.
#[cfg(target_family = "unix")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrimaryAtLookup {
    /// The worker's lease reserves the digest, but nothing put it back.
    ReservedAbsent,
    /// The worker leased the digest and uploaded it again.
    Reinserted,
}

/// An eviction takes `digest`'s entry out of the map, but its
/// executable-variant cleanup waits, before taking the variant lease, behind
/// an earlier entry's cleanup. A worker then leases the digest, resolves its
/// executable source and only later hardlinks it, after the old cleanup has
/// run. The hardlink must find the variant: a source the worker accepted is
/// one the old cleanup leaves in place.
#[cfg(target_family = "unix")]
async fn check_old_cleanup_before_variant_lease(
    lookup: ExecutableLookup,
    primary: PrimaryAtLookup,
) -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;

    const EARLIER: &str = "earlier-01";
    const EXECUTABLE: &str = "executable";
    const NEWCOMER: &str = "newcomer-entry";
    const EARLIER_HASH: &str = "0123456789abcdef000000000000000000030000000000000123456789abcdef";
    const NEWCOMER_HASH: &str = "0123456789abcdef000000000000000000040000000000000123456789abcdef";

    let content_path = make_temp_path("content_path");
    // With one-byte blocks the map holds exactly the blob sizes. The earlier
    // and executable entries fit together; the newcomer's insertion evicts
    // the earlier entry and then the executable one, in that order, in one
    // removal.
    let max_bytes = EARLIER.len() + EXECUTABLE.len() + 1;
    assert!(EXECUTABLE.len() + NEWCOMER.len() >= max_bytes);
    assert!(NEWCOMER.len() < max_bytes);
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        block_size: 1,
        eviction_policy: Some(EvictionPolicy {
            max_bytes,
            ..Default::default()
        }),
        ..Default::default()
    })
    .await?;
    let earlier = DigestInfo::try_new(EARLIER_HASH, EARLIER.len())?;
    let digest = DigestInfo::try_new(HASH1, EXECUTABLE.len())?;
    let newcomer = DigestInfo::try_new(NEWCOMER_HASH, NEWCOMER.len())?;
    let key = StoreKey::Digest(digest);

    let entered = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    store
        .clone()
        .register_remove_callback(Arc::new(CleanupGate {
            key: earlier,
            entered: entered.clone(),
            release: release.clone(),
        }))?;

    store.update_oneshot(earlier, EARLIER.into()).await?;
    store.update_oneshot(digest, EXECUTABLE.into()).await?;
    let variant = store.get_executable_hardlink_source(&digest).await?;
    assert!(Path::new(&variant).exists());

    let eviction = background_spawn!("evicting_insertion", {
        let store = store.clone();
        async move { store.update_oneshot(newcomer, NEWCOMER.into()).await }
    });
    drop(
        tokio::time::timeout(TASK_JOIN_LIMIT, entered.acquire())
            .await
            .expect("the eviction never reached the earlier entry's cleanup"),
    );
    assert!(!store.get_evicting_map().is_resident(&key));
    assert_eq!(
        store.executable_variant_lease_claims_for_test(&digest),
        0,
        "the old cleanup must not have taken the variant lease yet"
    );
    assert!(
        Path::new(&variant).exists(),
        "the old cleanup has deleted nothing yet"
    );

    store.lease_digest(&digest);
    if primary == PrimaryAtLookup::Reinserted {
        store.update_oneshot(digest, EXECUTABLE.into()).await?;
    }
    let (source, accepted_at_first_lookup) =
        resolve_executable_like_worker(&store, lookup, digest, EXECUTABLE).await?;
    assert_eq!(source, variant);
    info!(
        ?lookup,
        ?primary,
        accepted_at_first_lookup,
        "resolved executable source before the old cleanup ran"
    );

    // The old cleanup now runs to completion before the worker hardlinks.
    release.add_permits(1);
    join_bounded(eviction).await?;

    let input = format!("{content_path}-input");
    let linked = fs::hard_link(&source, &input).await;
    assert!(
        linked.is_ok(),
        "{lookup:?}/{primary:?}: the worker accepted {} at first lookup: \
         {accepted_at_first_lookup}, but the old cleanup deleted it: {linked:?}",
        Path::new(&source).display()
    );
    assert_eq!(
        read_file_contents(OsStr::new(&input)).await?,
        EXECUTABLE.as_bytes()
    );
    let mode = std::fs::metadata(&input)?.permissions().mode();
    assert_eq!(mode & 0o777, 0o555, "executable input must be 0o555");
    assert!(
        store.get_evicting_map().is_resident(&key),
        "the leased digest the worker linked from is resident"
    );
    store.release_digests(&[digest]).await;
    Ok(())
}

#[cfg(target_family = "unix")]
#[nativelink_test]
async fn batch_lookup_of_reserved_absent_digest_survives_old_cleanup() -> Result<(), Error> {
    check_old_cleanup_before_variant_lease(ExecutableLookup::Batch, PrimaryAtLookup::ReservedAbsent)
        .await
}

#[cfg(target_family = "unix")]
#[nativelink_test]
async fn single_lookup_of_reserved_absent_digest_survives_old_cleanup() -> Result<(), Error> {
    check_old_cleanup_before_variant_lease(
        ExecutableLookup::Single,
        PrimaryAtLookup::ReservedAbsent,
    )
    .await
}

#[cfg(target_family = "unix")]
#[nativelink_test]
async fn batch_lookup_of_reinserted_digest_survives_old_cleanup() -> Result<(), Error> {
    check_old_cleanup_before_variant_lease(ExecutableLookup::Batch, PrimaryAtLookup::Reinserted)
        .await
}

#[cfg(target_family = "unix")]
#[nativelink_test]
async fn single_lookup_of_reinserted_digest_survives_old_cleanup() -> Result<(), Error> {
    check_old_cleanup_before_variant_lease(ExecutableLookup::Single, PrimaryAtLookup::Reinserted)
        .await
}

/// Control for the tests above: when the old cleanup already holds the
/// variant lease, a worker's lookup of the reserved but absent digest does not
/// take the warm path. It populates the digest, and the cleanup, admitted
/// afterwards, keeps the variant of the now-resident digest.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn lookup_of_reserved_absent_digest_waits_for_cleanup_holding_variant_lease()
-> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    let variant = store.get_executable_hardlink_source(&digest).await?;
    let (admission, removal) = start_delayed_removal(&store, digest).await?;
    assert_eq!(store.executable_variant_lease_claims_for_test(&digest), 1);

    store.lease_digest(&digest);
    let first = store.get_executable_hardlink_sources(&[digest]).await;
    assert!(
        first[0].is_err(),
        "a digest whose variant lease is held is not a warm hit: {first:?}"
    );
    store.update_oneshot(digest, VALUE1.into()).await?;
    let resolve = background_spawn!("resolve_after_populate", {
        let store = store.clone();
        async move { store.get_executable_hardlink_source(&digest).await }
    });
    drop(admission);
    assert!(join_bounded(removal).await);
    let source = join_bounded(resolve).await?;
    assert_eq!(source, variant);
    hardlink_executable_input(&source, &format!("{content_path}-input")).await?;
    store.release_digests(&[digest]).await;
    Ok(())
}

/// This test simulates a full disk without needing one. It writes past the `RLIMIT_FSIZE`
/// cap, thus failing with `EFBIG`, which tokio defers exactly like `ENOSPC`. The
/// [`SIGXFSZ`](`libc::SIGXFSZ`) signal must be ignored or the kernel will kill the process
/// instead of failing the write.
///
/// SAFETY: [`SIG_IGN`](`libc::SIG_IGN`) is process-wide but no other test relies on
/// [`SIGXFSZ`](`libc::SIGXFSZ`).
#[cfg(unix)]
#[nativelink_test]
async fn deferred_write_error_does_not_emplace_truncated_file() -> Result<(), Error> {
    use rlimit::Resource;

    const FILE_SIZE_LIMIT: u64 = 1024 * 1024;

    let content_path = make_temp_path("content_path");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let data = vec![0u8; 2 * 1024 * 1024]; // 2MiB
    let digest = DigestInfo::try_new(&"aa".repeat(32), data.len())?;

    unsafe { libc::signal(libc::SIGXFSZ, libc::SIG_IGN) };
    let (old_soft, hard) = Resource::FSIZE.get()?;
    Resource::FSIZE.set(FILE_SIZE_LIMIT, hard)?;

    let result = store.update_oneshot(digest, data.into()).await;
    Resource::FSIZE.set(old_soft, hard)?;

    assert!(result.is_err(), "deferred write error must surface");

    let ghosts = std::fs::read_dir(format!("{content_path}/{DIGEST_FOLDER_V2}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<String>>();
    assert!(
        ghosts.is_empty(),
        "no file may reach the content path, found {ghosts:?}"
    );
    Ok(())
}

async fn setup_store_for_duplicates()
-> Result<(DigestInfo, core::pin::Pin<Box<Arc<FilesystemStore>>>), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;

    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest, VALUE1.into()).await?;
    Ok((digest, store))
}

#[nativelink_test]
async fn duplicate_upload_of_same_entry() -> Result<(), Error> {
    let (digest, store) = setup_store_for_duplicates().await?;

    let entry = store.get_file_entry_for_digest(&digest).await?;

    assert!(
        check_duplicate_files(&store.get_evicting_map(), &StoreKey::Digest(digest), &entry).await?,
        "Expected duplicate"
    );
    assert!(logs_contain(
        "Tried to check duplicate of an entry we already have!"
    ));
    Ok(())
}

#[nativelink_test]
async fn detect_duplicate_upload_different_size() -> Result<(), Error> {
    let (digest, store) = setup_store_for_duplicates().await?;

    let key = &StoreKey::Digest(digest);
    let temp_key = make_temp_key(key);
    let (entry, _temp_file, _temp_full_path) = store.make_temp_file(temp_key).await?;

    assert!(
        !check_duplicate_files(&store.get_evicting_map(), key, &Arc::new(entry)).await?,
        "Expected non-duplicate"
    );
    assert!(logs_contain(
        "Different data sizes, so non-duplicate entry_data_size=0 existing_data_size=10"
    ));
    Ok(())
}

#[nativelink_test]
async fn detect_duplicate_upload() -> Result<(), Error> {
    let (digest, store) = setup_store_for_duplicates().await?;

    let key = &StoreKey::Digest(digest);
    let temp_key = make_temp_key(key);
    let (mut entry, mut temp_file, temp_full_path) = store.make_temp_file(temp_key).await?;
    info!(?temp_full_path, "Temp full path");
    let mut data = Bytes::from_static(VALUE1.as_bytes());
    temp_file.write_all_buf(&mut data).await?;
    temp_file.flush().await?;
    *entry.data_size_mut() = 10;

    let arc_entry = Arc::new(entry);
    assert!(
        check_duplicate_files(&store.get_evicting_map(), key, &arc_entry).await?,
        "Expected duplicate"
    );
    assert!(logs_contain(
        "Identical files, so don't need to edit, skipping emplace"
    ));
    // Keep it alive until here to avoid early drop and delete, which breaks the test
    drop(arc_entry);
    Ok(())
}

#[nativelink_test]
async fn detect_same_key_different_contents() -> Result<(), Error> {
    let (digest, store) = setup_store_for_duplicates().await?;

    let key = &StoreKey::Digest(digest);
    let temp_key = make_temp_key(key);
    let (mut entry, mut temp_file, _temp_full_path) = store.make_temp_file(temp_key).await?;
    let mut data = Bytes::from_static(VALUE2.as_bytes());
    temp_file.write_all_buf(&mut data).await?;
    temp_file.flush().await?;
    *entry.data_size_mut() = 10;

    assert!(
        !check_duplicate_files(&store.get_evicting_map(), key, &Arc::new(entry)).await?,
        "Expected non-duplicate"
    );
    assert!(logs_contain("Files are different, so non-duplicate"));
    Ok(())
}

/// An upload must repair a stale index entry instead of failing the duplicate
/// comparison with ENOENT. Cover both digest and arbitrary string keys.
#[nativelink_test]
async fn upload_self_heals_map_disk_divergence() -> Result<(), Error> {
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        block_size: 1,
        ..Default::default()
    })
    .await?;
    for key in [
        StoreKey::Digest(DigestInfo::try_new(HASH1, VALUE1.len())?),
        StoreKey::new_str("upload-with-missing-file-123"),
    ] {
        store.update_oneshot(key.borrow(), VALUE1.into()).await?;
        let stale_entry = store.get_evicting_map().get(&key).await.unwrap();
        stale_entry
            .get_file_path_locked(|path| async move { fs::remove_file(path).await })
            .await?;
        assert_eq!(store.has(key.borrow()).await?, Some(VALUE1.len() as u64));

        store.update_oneshot(key.borrow(), VALUE1.into()).await?;
        let replacement = store.get_evicting_map().get(&key).await.unwrap();
        assert!(!Arc::ptr_eq(&stale_entry, &replacement));
        assert_eq!(
            store.get_part_unchunked(key, 0, None).await?,
            VALUE1.as_bytes(),
        );
    }
    assert!(logs_contain("Map/disk divergence in check_duplicate_files"));
    Ok(())
}

#[nativelink_test]
async fn stale_read_does_not_remove_replacement_upload() -> Result<(), Error> {
    static READ_FAILED: Semaphore = Semaphore::const_new(0);
    static RESUME_READ: Semaphore = Semaphore::const_new(0);

    struct GatedReadHooks;
    impl FileEntryHooks for GatedReadHooks {
        async fn on_read_error() {
            READ_FAILED.add_permits(1);
            RESUME_READ.acquire().await.unwrap().forget();
        }
    }

    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let store = FilesystemStore::<TestFileEntry<GatedReadHooks>>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    store
        .get_file_entry_for_digest(&digest)
        .await?
        .get_file_path_locked(|path| async move { fs::remove_file(path).await })
        .await?;
    let reading_store = store.clone();
    let read = spawn!("stale_read", async move {
        reading_store.get_part_unchunked(digest, 0, None).await
    });
    tokio::time::timeout(Duration::from_secs(5), READ_FAILED.acquire())
        .await
        .expect("read never reached the missing file")
        .unwrap()
        .forget();

    store.update_oneshot(digest, VALUE1.into()).await?;
    let replacement = store.get_file_entry_for_digest(&digest).await?;
    RESUME_READ.add_permits(1);
    assert_eq!(read.await.unwrap().unwrap_err().code, Code::NotFound);
    assert!(Arc::ptr_eq(
        &replacement,
        &store.get_file_entry_for_digest(&digest).await?,
    ));
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        VALUE1.as_bytes(),
    );
    Ok(())
}

#[nativelink_test]
async fn duplicate_comparison_checks_every_byte_and_empty_files() -> Result<(), Error> {
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let key = StoreKey::new_str("duplicate-boundaries");
    // The old loop underflowed for empty files and skipped the last byte for
    // files whose length was one more than a full comparison chunk.
    for size in [0, 1, 16 * 1024 + 1] {
        let original = Bytes::from(vec![b'a'; size]);
        store.update_oneshot(key.borrow(), original.clone()).await?;
        store.update_oneshot(key.borrow(), original.clone()).await?;
        let mut replacement = original.to_vec();
        if let Some(last) = replacement.last_mut() {
            *last = b'b';
        }
        let replacement = Bytes::from(replacement);
        store
            .update_oneshot(key.borrow(), replacement.clone())
            .await?;
        assert_eq!(
            store.get_part_unchunked(key.borrow(), 0, None).await?,
            replacement
        );
    }
    Ok(())
}

#[nativelink_test]
async fn failed_publish_preserves_previous_generation() -> Result<(), Error> {
    static FAIL_RENAME: AtomicBool = AtomicBool::new(false);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        if FAIL_RENAME.load(Ordering::Acquire) {
            return Err(std::io::Error::other("publish blocked by test"));
        }
        std::fs::rename(from, to)
    }

    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    let key = StoreKey::new_str("replace-me");
    store.update_oneshot(key.borrow(), VALUE1.into()).await?;
    FAIL_RENAME.store(true, Ordering::Release);
    assert!(
        store
            .update_oneshot(key.borrow(), VALUE2.into())
            .await
            .is_err()
    );
    assert_eq!(
        store.get_part_unchunked(key, 0, None).await?,
        VALUE1.as_bytes()
    );
    Ok(())
}

#[nativelink_test]
async fn older_upload_finishing_last_does_not_replace_newer_generation() -> Result<(), Error> {
    static STARTED: Semaphore = Semaphore::const_new(0);
    struct StartedHooks;
    impl FileEntryHooks for StartedHooks {
        async fn on_make_and_open(_path: &EncodedFilePath) -> Result<(), Error> {
            STARTED.add_permits(1);
            Ok(())
        }
    }

    let content_path = make_temp_path("content_path");
    let store = FilesystemStore::<TestFileEntry<StartedHooks>>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let (mut tx, rx) = make_buf_channel_pair();
    let older_store = store.clone();
    let upload = spawn!("older_upload", async move {
        older_store
            .update("key", rx, UploadSizeInfo::ExactSize(VALUE1.len() as u64))
            .await
    });
    tx.send(VALUE1.into()).await?;
    tokio::time::timeout(Duration::from_secs(5), STARTED.acquire())
        .await
        .expect("first upload never started")
        .unwrap()
        .forget();
    store.update_oneshot("key", VALUE2.into()).await?;
    tx.send_eof()?;
    upload.await.unwrap()?;
    assert_eq!(
        store.get_part_unchunked("key", 0, None).await?,
        VALUE2.as_bytes()
    );
    assert_eq!(
        list_file_names(&format!("{content_path}/{STR_FOLDER_V2}")).await?,
        vec!["key-2"],
        "the rejected generation must be retired from the content directory",
    );
    Ok(())
}

// Drop root capabilities to read all files so we can write tests about permissions
#[cfg(target_os = "linux")]
fn drop_all_fs_read_caps() -> Result<(), Error> {
    use caps::{CapSet, Capability};

    debug!(
        effective = ?caps::read(None, CapSet::Effective)?,
        permitted = ?caps::read(None, CapSet::Permitted)?,
        "Capabilities"
    );
    if caps::has_cap(None, CapSet::Permitted, Capability::CAP_DAC_READ_SEARCH)? {
        caps::drop(None, CapSet::Effective, Capability::CAP_DAC_READ_SEARCH)?;
    }
    if caps::has_cap(None, CapSet::Permitted, Capability::CAP_DAC_OVERRIDE)? {
        caps::drop(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE)?;
    }
    Ok(())
}

// There are no capabilities to drop off Linux; a non-root run already gets
// permission errors, which is all these tests need.
#[cfg(all(unix, not(target_os = "linux")))]
const fn drop_all_fs_read_caps() -> Result<(), Error> {
    Ok(())
}

#[cfg(unix)]
#[nativelink_test]
async fn duplicate_comparison_propagates_permission_errors() -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;

    drop_all_fs_read_caps()?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    debug!("Did first update");
    let entry = store.get_file_entry_for_digest(&digest).await?;
    entry
        .get_file_path_locked(|path| async move {
            debug!("Locked down {path:?} permissions");
            let res = fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).await;
            debug!(
                "{path:?} permissions are now {:?}",
                fs::metadata(&path).await?.permissions()
            );
            res
        })
        .await?;
    let result = store.update_oneshot(digest, VALUE1.into()).await;
    debug!("Checked {digest}, got {result:?}");
    entry
        .get_file_path_locked(|path| async move {
            debug!("Opened up {path:?} permissions");
            let res = fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).await;
            debug!(
                "{path:?} permissions are now {:?}",
                fs::metadata(&path).await?.permissions()
            );
            res
        })
        .await?;
    assert_eq!(result.unwrap_err().code, Code::PermissionDenied);
    assert!(Arc::ptr_eq(
        &entry,
        &store.get_file_entry_for_digest(&digest).await?
    ));
    Ok(())
}

/// Map/disk divergence (map says present, file is gone) must surface as a
/// recoverable warn and remove the stale entry — not a fatal error.
#[nativelink_test]
async fn get_part_on_map_disk_divergence_warns_and_removes_entry() -> Result<(), Error> {
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let content_path = make_temp_path("content_path");
    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: make_temp_path("temp_path"),
            eviction_policy: None,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest, VALUE1.into()).await?;

    // Delete the backing file out from under the still-present map entry.
    let content_file = content_file_path(&content_path, &StoreKey::Digest(digest)).await?;
    fs::remove_file(&content_file).await?;

    let err = store
        .get_part_unchunked(digest, 0, None)
        .await
        .expect_err("divergent read must fail");
    assert_eq!(
        err.code,
        Code::NotFound,
        "divergent read must surface NotFound"
    );

    assert!(
        logs_contain("Filesystem store map/disk divergence"),
        "divergence must be logged as a recoverable warn"
    );
    assert!(
        !logs_contain("process probably need restarted"),
        "stale fatal-sounding message must be gone"
    );
    assert_eq!(
        store.has(digest).await?,
        None,
        "stale entry must be removed from the map"
    );

    Ok(())
}

/// unref must be idempotent when the file is already gone: the entry is
/// marked Temp so a second unref early-returns instead of racing the
/// vanished path again.
#[nativelink_test]
async fn unref_is_idempotent_when_file_already_gone() -> Result<(), Error> {
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let content_path = make_temp_path("content_path");
    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: make_temp_path("temp_path"),
            eviction_policy: None,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest, VALUE1.into()).await?;
    let file_entry = store.get_file_entry_for_digest(&digest).await?;

    let content_file = content_file_path(&content_path, &StoreKey::Digest(digest)).await?;
    fs::remove_file(&content_file).await?;

    // First unref: rename hits ENOENT (benign) and flips the entry to Temp.
    file_entry.unref().await;
    // Second unref: must hit the Temp early-return, proving the flip stuck.
    file_entry.unref().await;

    assert!(
        logs_contain("File is already a temp file"),
        "second unref should early-return as a Temp file (idempotent)"
    );

    Ok(())
}

/// rename ENOENT is ambiguous: a missing temp directory must not be mistaken
/// for a vanished source. With the source still present, unref must warn and
/// leave the content file intact rather than flip to Temp and orphan it.
#[nativelink_test]
async fn unref_does_not_orphan_content_file_when_temp_dir_missing() -> Result<(), Error> {
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let store = Box::pin(
        FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: None,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest, VALUE1.into()).await?;
    let file_entry = store.get_file_entry_for_digest(&digest).await?;

    // Remove the temp dir so rename's destination parent is gone (ENOENT)
    // while the source content file is perfectly intact.
    fs::remove_dir_all(format!("{temp_path}/{DIGEST_FOLDER_V2}")).await?;

    file_entry.unref().await;

    assert!(
        logs_contain("Failed to rename file"),
        "missing temp dir (source present) must warn, not be treated as benign"
    );
    assert!(
        !logs_contain("treating as benign"),
        "an intact content file must not take the benign vanished-source path"
    );
    // The content file must still exist — not orphaned by a wrong Temp flip.
    let content_file = content_file_path(&content_path, &StoreKey::Digest(digest)).await?;
    assert_eq!(
        read_file_contents(&content_file).await?,
        VALUE1.as_bytes(),
        "content file must remain intact after a failed unref rename"
    );

    Ok(())
}

// Exercises the macOS flush coalescer's round machinery under a burst of
// concurrent uploads (on other platforms this is a plain concurrency smoke
// test). Two hundred small, independently durable blobs are used because each
// upload would otherwise issue its own multi-millisecond `F_FULLFSYNC` on
// macOS. The data volume is deliberately small so file I/O does not hide the
// synchronization behavior: every blob must complete and survive restart,
// while the device-wide barrier count must remain far below the blob count.
#[nativelink_test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_upload_burst_round_trip_test() -> Result<(), Error> {
    const NUM_BLOBS: u64 = 200;
    let content_path = make_temp_path("content_path_burst");
    let temp_path = make_temp_path("temp_path_burst");
    let spec = FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        eviction_policy: None,
        block_size: 1,
        ..Default::default()
    };
    let make_digest = |i: u64| {
        let mut hash = [0u8; 32];
        hash[0] = 0xbb;
        hash[1..9].copy_from_slice(&i.to_le_bytes());
        DigestInfo::new(hash, 25)
    };
    {
        let filesystem_store = FilesystemStore::<FileEntryImpl>::new(&spec).await?;
        let store = Store::new(filesystem_store.clone());
        let mut handles = Vec::new();
        for i in 0..NUM_BLOBS {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                let content = format!("burst-content-{i:011}");
                store
                    .update_oneshot(make_digest(i), content.into_bytes().into())
                    .await
            }));
        }
        for handle in handles {
            handle.await.expect("upload task panicked")?;
        }
        #[cfg(target_os = "macos")]
        {
            let full_flush_count = filesystem_store
                .full_flush_count_for_test()
                .expect("writable test store must have a flush coalescer");
            assert!(
                full_flush_count > 0 && full_flush_count < NUM_BLOBS / 2,
                "{NUM_BLOBS} concurrent uploads should share device-wide barriers; observed {full_flush_count} F_FULLFSYNC rounds"
            );
        }
        for i in 0..NUM_BLOBS {
            assert_eq!(
                store.get_part_unchunked(make_digest(i), 0, None).await?,
                format!("burst-content-{i:011}").as_bytes(),
                "round trip failed for blob {i}"
            );
        }
    }
    // A fresh store over the same paths must find every published blob.
    let store = Store::new(FilesystemStore::<FileEntryImpl>::new(&spec).await?);
    for i in 0..NUM_BLOBS {
        assert_eq!(
            store.has(make_digest(i)).await?,
            Some(25),
            "blob {i} missing after restart"
        );
    }
    Ok(())
}

/// Regression test. Operations on the eviction map (add/removal) can run at different times
/// w.r.t. moving files on a filesystem. In the past, file names contained only the digest and size.
///
/// The following happened in past.
/// - An entry was evicted from the map. The removal fs task was scheduled.
/// - Meanwhile, a client reuploads the file. A new entry is added to the map, and the file is moved
///   into the cache on the filesystem.
/// - Now the fs removal task runs, removing the file on the filesystem, leaving an entry in the map.
///
/// This was mitigated by making file names unique using a generation counter.
#[nativelink_test]
async fn eviction_unref_blocked_in_the_drain_does_not_hold_the_writer() -> Result<(), Error> {
    // Before the retirement drain, the writer retired what its insertion
    // evicted and waited here; now it only queues that retirement.
    static ENTERED: Semaphore = Semaphore::const_new(0);
    static RELEASE: Semaphore = Semaphore::const_new(0);
    struct Hooks;
    impl FileEntryHooks for Hooks {
        async fn on_unref_async<Fe: FileEntry>(entry: &Fe) {
            if entry.len() == VALUE1.len() as u64 {
                ENTERED.add_permits(1);
                let permit = RELEASE.acquire().await.unwrap();
                permit.forget();
            }
        }
    }
    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let other = "y".repeat(64);
    let digest2 = DigestInfo::try_new(HASH2, other.len())?;
    let store = Arc::new(
        FilesystemStore::<TestFileEntry<Hooks>>::new(&FilesystemSpec {
            content_path: make_temp_path("diagnostic-content"),
            temp_path: make_temp_path("diagnostic-temp"),
            eviction_policy: Some(EvictionPolicy {
                max_count: 1,
                ..Default::default()
            }),
            block_size: 1,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest1, VALUE1.into()).await?;
    let writer = store.clone();
    let task = tokio::spawn(async move { writer.update_oneshot(digest2, other.into()).await });
    tokio::time::timeout(Duration::from_secs(5), ENTERED.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the writer waited for the unref of the entry it evicted")
        .unwrap()?;
    assert_eq!(store.has(digest2).await?, Some(digest2.size_bytes()));
    assert_eq!(store.has(digest1).await?, None);
    assert!(store.retirement_accounting().draining);
    RELEASE.add_permits(1);
    tokio::time::timeout(Duration::from_secs(5), store.retirement_settled_for_test())
        .await
        .expect("the drain did not finish");
    Ok(())
}

#[nativelink_test]
async fn eviction_whose_entry_path_is_held_is_detached_from_the_drain() -> Result<(), Error> {
    static ENTERED: Semaphore = Semaphore::const_new(0);
    struct Hooks;
    impl FileEntryHooks for Hooks {
        async fn on_unref_async<Fe: FileEntry>(_entry: &Fe) {
            ENTERED.add_permits(1);
        }
    }
    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let other = "z".repeat(64);
    let digest2 = DigestInfo::try_new(HASH2, other.len())?;
    let store = Arc::new(
        FilesystemStore::<TestFileEntry<Hooks>>::new(&FilesystemSpec {
            content_path: make_temp_path("diagnostic-lock-content"),
            temp_path: make_temp_path("diagnostic-lock-temp"),
            eviction_policy: Some(EvictionPolicy {
                max_count: 1,
                ..Default::default()
            }),
            block_size: 1,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest1, VALUE1.into()).await?;
    let entry = store.get_file_entry_for_digest(&digest1).await?;
    let held = entry.get_encoded_file_path().read().await;
    tokio::time::timeout(
        Duration::from_secs(5),
        store.update_oneshot(digest2, other.into()),
    )
    .await
    .expect("the writer waited for a reader of the entry it evicted")?;
    // The drain handed the held entry to a detached unref, which entered the
    // unref and now waits for the reader.
    tokio::time::timeout(Duration::from_secs(5), ENTERED.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let accounting = store.retirement_accounting();
            if !accounting.draining && accounting.queued_units == 0 {
                return accounting;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .map(|accounting| assert_eq!(accounting.detached_in_flight, 1, "{accounting:?}"))
    .expect("the drain waited behind a reader");
    drop(held);
    tokio::time::timeout(Duration::from_secs(5), store.retirement_settled_for_test())
        .await
        .expect("the detached unref did not finish");
    assert_eq!(store.retirement_accounting().detached_total, 1);
    assert_eq!(store.has(digest2).await?, Some(digest2.size_bytes()));
    Ok(())
}

#[nativelink_test]
async fn concurrent_evicting_writes_return_while_their_retirements_wait() -> Result<(), Error> {
    static ENTERED: Semaphore = Semaphore::const_new(0);
    static RELEASE: Semaphore = Semaphore::const_new(0);
    struct Hooks;
    impl FileEntryHooks for Hooks {
        async fn on_unref_async<Fe: FileEntry>(entry: &Fe) {
            ENTERED.add_permits(1);
            if entry.len() == 20 {
                RELEASE.acquire().await.unwrap().forget();
            }
        }
    }
    let first = DigestInfo::try_new(HASH1, 10)?;
    let second = DigestInfo::try_new(HASH2, 20)?;
    let third = DigestInfo::try_new("a".repeat(64).as_str(), 30)?;
    let fourth = DigestInfo::try_new("b".repeat(64).as_str(), 40)?;
    let store = Arc::new(
        FilesystemStore::<TestFileEntry<Hooks>>::new(&FilesystemSpec {
            content_path: make_temp_path("diagnostic-concurrent-content"),
            temp_path: make_temp_path("diagnostic-concurrent-temp"),
            eviction_policy: Some(EvictionPolicy {
                max_count: 2,
                ..Default::default()
            }),
            block_size: 1,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(first, "1".repeat(10).into()).await?;
    store.update_oneshot(second, "2".repeat(20).into()).await?;
    let entry = store.get_file_entry_for_digest(&first).await?;
    // Fetching first makes it MRU: touch second to ensure first is evicted first.
    store.get_file_entry_for_digest(&second).await?;
    let held = entry.get_encoded_file_path().read().await;
    // The first write evicts `first`, whose reader holds its path; the second
    // evicts `second`, whose unref blocks in the drain. Neither write waits.
    let writes = futures::future::try_join(
        {
            let store = store.clone();
            async move { store.update_oneshot(third, "3".repeat(30).into()).await }
        },
        {
            let store = store.clone();
            async move { store.update_oneshot(fourth, "4".repeat(40).into()).await }
        },
    );
    tokio::time::timeout(Duration::from_secs(5), writes)
        .await
        .expect("an evicting write waited for its retirement")?;
    for _ in 0..2 {
        tokio::time::timeout(Duration::from_secs(5), ENTERED.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
    }
    assert_eq!(store.has(third).await?, Some(30));
    assert_eq!(store.has(fourth).await?, Some(40));
    drop(held);
    RELEASE.add_permits(1);
    tokio::time::timeout(Duration::from_secs(5), store.retirement_settled_for_test())
        .await
        .expect("retirement did not finish");
    drop(entry);
    Ok(())
}

#[nativelink_test]
async fn stale_unref_must_not_delete_reuploaded_file() -> Result<(), Error> {
    static UNREF_GATE: Semaphore = Semaphore::const_new(0);

    struct GatedUnrefHooks;
    impl FileEntryHooks for GatedUnrefHooks {
        async fn on_unref_async<Fe: FileEntry>(entry: &Fe) {
            if entry.len() == VALUE1.len() as u64 {
                let _permit = UNREF_GATE
                    .acquire()
                    .await
                    .expect("unref gate closed unexpectedly");
            }
        }
    }

    // Sized apart from VALUE1 so the hook can tell the two unrefs apart.
    let other_value = "y".repeat(64);
    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let digest2 = DigestInfo::try_new(HASH2, other_value.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Arc::new(
        FilesystemStore::<TestFileEntry<GatedUnrefHooks>>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: Some(EvictionPolicy {
                max_count: 1,
                ..Default::default()
            }),
            block_size: 1,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest1, VALUE1.into()).await?;
    assert!(
        content_file_exists(&content_path, &StoreKey::Digest(digest1)).await?,
        "digest1 should be on disk after its upload"
    );

    // Evict digest1, suspending its unref before the rename.
    let evicting_store = store.clone();
    let eviction = spawn!("stale_unref_eviction", async move {
        evicting_store
            .update_oneshot(digest2, other_value.into())
            .await
    });
    while store.has(digest1).await?.is_some() {
        tokio::task::yield_now().await;
    }
    assert!(
        content_file_exists(&content_path, &StoreKey::Digest(digest1)).await?,
        "digest1's file should outlive its map entry until unref runs"
    );

    // The client sees the blob as missing and re-uploads it.
    store.update_oneshot(digest1, VALUE1.into()).await?;
    assert_eq!(
        store.has(digest1).await?,
        Some(VALUE1.len() as u64),
        "the re-uploaded blob should be resident"
    );
    assert_eq!(
        store.get_part_unchunked(digest1, 0, None).await?,
        VALUE1.as_bytes(),
        "the re-uploaded blob should be readable"
    );

    // Let the evicted entry finish its unref.
    UNREF_GATE.add_permits(1);
    eviction
        .await
        .expect("eviction task panicked")
        .err_tip(|| "Failed to insert digest2")?;

    assert!(
        content_file_exists(&content_path, &StoreKey::Digest(digest1)).await?,
        "stale cleanup deleted the re-uploaded file for {digest1}"
    );
    assert_eq!(
        store.get_part_unchunked(digest1, 0, None).await?,
        VALUE1.as_bytes(),
        "the re-uploaded blob should still be readable after the stale unref"
    );
    Ok(())
}

/// Control for [`stale_unref_must_not_delete_reuploaded_file`]: the same
/// evict-then-reupload sequence, with a difference that `unref()` is allowed to
/// finish before the reupload rather than after it.
#[nativelink_test]
async fn reupload_after_completed_unref_is_readable() -> Result<(), Error> {
    static UNREF_GATE: Semaphore = Semaphore::const_new(0);

    struct GatedUnrefHooks;
    impl FileEntryHooks for GatedUnrefHooks {
        async fn on_unref_async<Fe: FileEntry>(entry: &Fe) {
            if entry.len() == VALUE1.len() as u64 {
                let _permit = UNREF_GATE
                    .acquire()
                    .await
                    .expect("unref gate closed unexpectedly");
            }
        }
    }

    let other_value = "y".repeat(64);
    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let digest2 = DigestInfo::try_new(HASH2, other_value.len())?;
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");

    let store = Arc::new(
        FilesystemStore::<TestFileEntry<GatedUnrefHooks>>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: Some(EvictionPolicy {
                max_count: 1,
                ..Default::default()
            }),
            block_size: 1,
            ..Default::default()
        })
        .await?,
    );
    store.update_oneshot(digest1, VALUE1.into()).await?;

    // Same eviction, drained fully before the client re-uploads.
    let evicting_store = store.clone();
    let eviction = spawn!("completed_unref_eviction", async move {
        evicting_store
            .update_oneshot(digest2, other_value.into())
            .await
    });
    while store.has(digest1).await?.is_some() {
        tokio::task::yield_now().await;
    }
    UNREF_GATE.add_permits(1);
    eviction
        .await
        .expect("eviction task panicked")
        .err_tip(|| "Failed to insert digest2")?;

    // No stale unref is outstanding by the time this lands.
    store.update_oneshot(digest1, VALUE1.into()).await?;
    assert!(
        content_file_exists(&content_path, &StoreKey::Digest(digest1)).await?,
        "the re-uploaded file should be on disk for {digest1}"
    );
    assert_eq!(
        store.get_part_unchunked(digest1, 0, None).await?,
        VALUE1.as_bytes(),
        "the re-uploaded blob should be readable"
    );
    Ok(())
}

/// Legacy `s/` entries, one per way `-` can appear in an arbitrary string key.
const LEGACY_STR_ENTRIES: [(&str, &str); 4] = [
    ("foo", "no hyphen"),
    ("foo-bar", "embedded hyphen"),
    ("foo-1", "numeric suffix"),
    ("foo-bar-12345", "hyphens and a numeric suffix"),
];

#[nativelink_test]
async fn upgrade_from_v1_folders_stamps_generation_on_all_keys() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    fs::create_dir_all(format!("{content_path}/{STR_FOLDER_V1}")).await?;
    fs::create_dir_all(format!("{content_path}/{DIGEST_FOLDER_V1}")).await?;

    for (key, value) in LEGACY_STR_ENTRIES {
        std::fs::write(format!("{content_path}/{STR_FOLDER_V1}/{key}"), value)
            .err_tip(|| format!("writing legacy string entry {key}"))?;
    }
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    std::fs::write(
        format!("{content_path}/{DIGEST_FOLDER_V1}/{digest}"),
        VALUE1,
    )?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        block_size: 1,
        ..Default::default()
    })
    .await?;

    assert_entries_readable(&store).await?;
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        Bytes::from(VALUE1),
        "the legacy digest entry should survive the upgrade"
    );

    assert_eq!(
        list_file_names(&format!("{content_path}/{STR_FOLDER_V2}")).await?,
        vec!["foo-0", "foo-1-0", "foo-bar-0", "foo-bar-12345-0"],
    );
    assert_eq!(
        list_file_names(&format!("{content_path}/{DIGEST_FOLDER_V2}")).await?,
        vec![format!("{digest}-0")],
    );
    for legacy_folder in [STR_FOLDER_V1, DIGEST_FOLDER_V1] {
        assert_eq!(
            list_file_names(&format!("{content_path}/{legacy_folder}")).await?,
            Vec::<String>::new(),
            "{legacy_folder} should be drained by a successful migration"
        );
    }

    Ok(())
}

/// The pre-Dec-2024 layout kept every blob directly under `content_path`, so
/// both migrations must run, in order.
#[nativelink_test]
async fn upgrade_from_flat_layout_moves_digests_then_stamps_generation() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    fs::create_dir_all(&content_path).await?;

    let digest1 = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let digest2 = DigestInfo::try_new(HASH2, VALUE2.len())?;
    std::fs::write(format!("{content_path}/{digest1}"), VALUE1)?;
    std::fs::write(format!("{content_path}/{digest2}"), VALUE2)?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        block_size: 1,
        ..Default::default()
    })
    .await?;

    assert_eq!(
        store.get_part_unchunked(digest1, 0, None).await?,
        Bytes::from(VALUE1),
    );
    assert_eq!(
        store.get_part_unchunked(digest2, 0, None).await?,
        Bytes::from(VALUE2),
    );

    let mut expected = vec![format!("{digest1}-0"), format!("{digest2}-0")];
    expected.sort();
    assert_eq!(
        list_file_names(&format!("{content_path}/{DIGEST_FOLDER_V2}")).await?,
        expected,
    );

    Ok(())
}

#[nativelink_test]
async fn upgraded_layout_is_not_migrated_again_on_restart() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    fs::create_dir_all(format!("{content_path}/{STR_FOLDER_V1}")).await?;
    fs::create_dir_all(format!("{content_path}/{DIGEST_FOLDER_V1}")).await?;

    for (key, value) in LEGACY_STR_ENTRIES {
        std::fs::write(format!("{content_path}/{STR_FOLDER_V1}/{key}"), value)?;
    }
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    std::fs::write(
        format!("{content_path}/{DIGEST_FOLDER_V1}/{digest}"),
        VALUE1,
    )?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        block_size: 1,
        ..Default::default()
    })
    .await?;
    assert_entries_readable(&store).await?;
    let after_upgrade = list_file_names(&format!("{content_path}/{STR_FOLDER_V2}")).await?;
    // Another test may keep a file open throughout these restarts.
    // Restarting this store must not wait for the process-wide counter.
    let unrelated_permit = fs::get_permit().await;
    drop(store);

    for restart in 1..=2 {
        let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            block_size: 1,
            ..Default::default()
        })
        .await?;
        assert_entries_readable(&store)
            .await
            .err_tip(|| format!("after restart {restart}"))?;
        assert_eq!(
            store.get_part_unchunked(digest, 0, None).await?,
            Bytes::from(VALUE1),
        );
        assert_eq!(
            list_file_names(&format!("{content_path}/{STR_FOLDER_V2}")).await?,
            after_upgrade,
            "restart {restart} renamed already-migrated files"
        );
        assert_eq!(
            list_file_names(&format!("{content_path}/{DIGEST_FOLDER_V2}")).await?,
            vec![format!("{digest}-0")],
        );
        drop(store);
    }

    drop(unrelated_permit);
    Ok(())
}

#[nativelink_test]
async fn generations_assigned_after_upgrade_survive_restart() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    fs::create_dir_all(format!("{content_path}/{STR_FOLDER_V1}")).await?;
    std::fs::write(format!("{content_path}/{STR_FOLDER_V1}/foo-1"), "legacy")?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        block_size: 1,
        ..Default::default()
    })
    .await?;
    store
        .update_oneshot(StoreKey::new_str("foo-1"), VALUE1.into())
        .await?;
    store
        .update_oneshot(StoreKey::new_str("bar"), VALUE2.into())
        .await?;
    drop(store);
    wait_for_no_open_files().await?;

    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        block_size: 1,
        ..Default::default()
    })
    .await?;
    assert_eq!(
        store
            .get_part_unchunked(StoreKey::new_str("foo-1"), 0, None)
            .await?,
        Bytes::from(VALUE1),
        "the rewritten entry should replace the migrated legacy one"
    );
    assert_eq!(
        store
            .get_part_unchunked(StoreKey::new_str("bar"), 0, None)
            .await?,
        Bytes::from(VALUE2),
    );
    assert_eq!(
        store.has(StoreKey::new_str("foo")).await?,
        None,
        "`foo-1` must never be read back as key `foo` with generation 1"
    );

    Ok(())
}

#[nativelink_test]
async fn startup_keeps_newest_generation_and_advances_counter() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let dir = format!("{content_path}/{STR_FOLDER_V2}");
    fs::create_dir_all(&dir).await?;
    // Simulate a crash with multiple generations left on disk. Use creation
    // order opposite to generation order to avoid depending on directory order.
    std::fs::write(format!("{dir}/key-100"), VALUE2)?;
    std::fs::write(format!("{dir}/key-3"), VALUE1)?;
    let spec = FilesystemSpec {
        content_path,
        temp_path,
        block_size: 1,
        ..Default::default()
    };
    let store = FilesystemStore::<FileEntryImpl>::new(&spec).await?;
    assert_eq!(
        store.get_part_unchunked("key", 0, None).await?,
        VALUE2.as_bytes()
    );
    assert_eq!(list_file_names(&dir).await?, vec!["key-100"]);
    store.update_oneshot("key", VALUE1.into()).await?;
    assert_eq!(list_file_names(&dir).await?, vec!["key-101"]);
    drop(store);
    wait_for_no_open_files().await?;
    let store = FilesystemStore::<FileEntryImpl>::new(&spec).await?;
    assert_eq!(
        store.get_part_unchunked("key", 0, None).await?,
        VALUE1.as_bytes()
    );
    Ok(())
}

#[nativelink_test]
async fn failed_flat_migration_remains_readable() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    fs::create_dir_all(&content_path).await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    std::fs::write(format!("{content_path}/{digest}"), VALUE1)?;
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path,
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        |_, _| Err(std::io::Error::other("migration blocked by test")),
    )
    .await?;
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        VALUE1.as_bytes()
    );
    Ok(())
}

#[nativelink_test]
async fn replacing_unmigrated_entry_retires_legacy_file() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let legacy_dir = format!("{content_path}/{STR_FOLDER_V1}");
    fs::create_dir_all(&legacy_dir).await?;
    let legacy_file = format!("{legacy_dir}/key");
    std::fs::write(&legacy_file, VALUE1)?;
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path,
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        |from, to| {
            if from.to_string_lossy().ends_with("/s/key") {
                return Err(std::io::Error::other("migration blocked by test"));
            }
            std::fs::rename(from, to)
        },
    )
    .await?;
    store.update_oneshot("key", VALUE2.into()).await?;
    assert_eq!(
        store.get_part_unchunked("key", 0, None).await?,
        VALUE2.as_bytes()
    );
    assert!(
        !Path::new(&legacy_file).exists(),
        "retired legacy content must not reappear after restart"
    );
    Ok(())
}

#[nativelink_test]
async fn partial_migration_stays_readable_and_retries_next_startup() -> Result<(), Error> {
    const STUCK_KEY: &str = "stuck";
    const MOVED_KEY: &str = "moved";
    static BLOCK_MIGRATION: AtomicBool = AtomicBool::new(true);

    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        if BLOCK_MIGRATION.load(Ordering::Acquire)
            && from
                .to_string_lossy()
                .ends_with(&format!("/{STR_FOLDER_V1}/{STUCK_KEY}"))
        {
            return Err(std::io::Error::other("migration blocked by test"));
        }
        std::fs::rename(from, to)
    }

    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    fs::create_dir_all(format!("{content_path}/{STR_FOLDER_V1}")).await?;
    std::fs::write(
        format!("{content_path}/{STR_FOLDER_V1}/{STUCK_KEY}"),
        VALUE1,
    )?;
    std::fs::write(
        format!("{content_path}/{STR_FOLDER_V1}/{MOVED_KEY}"),
        VALUE2,
    )?;

    let spec = FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        block_size: 1,
        ..Default::default()
    };

    let store =
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(&spec, rename_fn).await?;
    assert_eq!(
        store
            .get_part_unchunked(StoreKey::new_str(STUCK_KEY), 0, None)
            .await?,
        Bytes::from(VALUE1),
        "a file that failed to migrate must still be served from where it sits"
    );
    assert_eq!(
        store
            .get_part_unchunked(StoreKey::new_str(MOVED_KEY), 0, None)
            .await?,
        Bytes::from(VALUE2),
    );
    assert_eq!(
        list_file_names(&format!("{content_path}/{STR_FOLDER_V1}")).await?,
        vec![STUCK_KEY.to_string()],
    );
    assert_eq!(
        list_file_names(&format!("{content_path}/{STR_FOLDER_V2}")).await?,
        vec![format!("{MOVED_KEY}-0")],
    );
    drop(store);
    wait_for_no_open_files().await?;

    BLOCK_MIGRATION.store(false, Ordering::Release);
    let store =
        FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(&spec, rename_fn).await?;
    assert_eq!(
        list_file_names(&format!("{content_path}/{STR_FOLDER_V1}")).await?,
        Vec::<String>::new(),
        "the retry should drain the legacy folder"
    );
    assert_eq!(
        list_file_names(&format!("{content_path}/{STR_FOLDER_V2}")).await?,
        vec![format!("{MOVED_KEY}-0"), format!("{STUCK_KEY}-0")],
    );
    assert_eq!(
        store
            .get_part_unchunked(StoreKey::new_str(STUCK_KEY), 0, None)
            .await?,
        Bytes::from(VALUE1),
    );

    Ok(())
}

/// Invalid declared sizes must fail before duplicate detection can accept the
/// existing identical blob, and before finalization can publish a replacement.
#[nativelink_test]
async fn invalid_stream_size_does_not_accept_duplicate_or_replace_blob() -> Result<(), Error> {
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("stream_integrity_content"),
        temp_path: make_temp_path("stream_integrity_temp"),
        max_concurrent_writes: 1,
        ..Default::default()
    })
    .await?;
    let key = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(key, VALUE1.into()).await?;

    for value in [VALUE1, VALUE2] {
        for declared in [
            UploadSizeInfo::ExactSize(value.len() as u64 + 1),
            UploadSizeInfo::MaxSize(value.len() as u64 - 1),
        ] {
            let (mut tx, rx) = make_buf_channel_pair();
            let (sent, updated) = tokio::join!(
                async move {
                    tx.send(value.into()).await?;
                    tx.send_eof()
                },
                store.update(key, rx, declared)
            );
            sent?;
            assert_eq!(updated.unwrap_err().code, Code::InvalidArgument);
            assert_eq!(
                store.get_part_unchunked(key, 0, None).await?,
                VALUE1.as_bytes()
            );
        }
    }
    // Rejections must not consume the sole write permit or poison later writes.
    store.update_oneshot(key, VALUE2.into()).await?;
    assert_eq!(
        store.get_part_unchunked(key, 0, None).await?,
        VALUE2.as_bytes()
    );
    Ok(())
}

#[nativelink_test]
async fn incomplete_whole_file_does_not_replace_existing_blob() -> Result<(), Error> {
    let temp_path = make_temp_path("whole_file_integrity");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("whole_file_integrity_content"),
        temp_path: temp_path.clone(),
        block_size: 1,
        ..Default::default()
    })
    .await?;
    let key = DigestInfo::try_new(HASH1, 78)?;
    let good = Bytes::from(vec![7; 78]);
    store.update_oneshot(key, good.clone()).await?;
    let path = OsString::from(format!("{temp_path}/incomplete"));
    let file = fs::create_file(&path).await?;
    let error = store
        .update_with_whole_file(key, path, file, UploadSizeInfo::ExactSize(78))
        .await
        .unwrap_err();
    assert_eq!(error.code, Code::InvalidArgument);
    assert_eq!(store.get_part_unchunked(key, 0, None).await?, good);
    Ok(())
}

/// Writes `value` through the streaming `update` entry point.
#[cfg(target_family = "unix")]
async fn stream_update(
    store: &FilesystemStore<FileEntryImpl>,
    digest: DigestInfo,
    value: &'static str,
) -> Result<u64, Error> {
    let (mut tx, rx) = make_buf_channel_pair();
    let (sent, updated) = tokio::join!(
        async move {
            tx.send(value.into()).await?;
            tx.send_eof()
        },
        store.update(digest, rx, UploadSizeInfo::ExactSize(value.len() as u64))
    );
    sent?;
    updated
}

/// Waits until no temporary file is left in `temp_path`. A failed write
/// removes its temporary file from a background task, so this polls, within
/// a bound.
#[cfg(target_family = "unix")]
async fn wait_for_empty_temp_dir(temp_path: &str) -> Result<(), Error> {
    let dir = format!("{temp_path}/{DIGEST_FOLDER_V2}");
    for _ in 0..10_000 {
        if list_file_names(&dir).await?.is_empty() {
            return Ok(());
        }
        sleep(Duration::from_millis(1)).await;
    }
    Err(make_err!(
        Code::DeadlineExceeded,
        "Temporary files remain in {dir}: {:?}",
        list_file_names(&dir).await?
    ))
}

/// `max_concurrent_writes` bounds each write's fsync and rename, so with a
/// limit of one a second write must not reach its rename until the first
/// write's rename has really finished. The rename runs in a task that
/// outlives a cancelled caller, so when `cancel_first` is set the first
/// caller is cancelled while its rename is held: that task, not the caller,
/// must keep the write permit.
#[cfg(target_family = "unix")]
async fn check_write_limit_spans_rename(
    gate: &'static RenameGate,
    rename_fn: fn(&OsStr, &OsStr) -> Result<(), std::io::Error>,
    cancel_first: bool,
) -> Result<(), Error> {
    let _release = ReleaseAllOnDrop(gate);
    let temp_path = make_temp_path("temp_path");
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: temp_path.clone(),
            max_concurrent_writes: 1,
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    let first_digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let second_digest = DigestInfo::try_new(HASH2, VALUE2.len())?;

    // The first write streams its data, takes the only permit and holds in
    // its rename.
    let first = background_spawn!("write_limit_first", {
        let store = store.clone();
        async move { stream_update(&store, first_digest, VALUE1).await }
    });
    gate.wait_entered(1).await?;
    let first = if cancel_first {
        // Awaiting the aborted task proves its future has been dropped.
        first.abort();
        let cancelled = tokio::time::timeout(TASK_JOIN_LIMIT, first)
            .await
            .expect("cancelled task did not finish within TASK_JOIN_LIMIT");
        assert!(cancelled.is_err_and(|e| e.is_cancelled()));
        None
    } else {
        Some(first)
    };

    // While it waits for the permit, the second write holds an open temp
    // file, which tests that wait for every file in the process to close
    // count; keep the wait short.
    let mut second = background_spawn!("write_limit_second", {
        let store = store.clone();
        async move { store.update_oneshot(second_digest, VALUE2.into()).await }
    });
    if let Ok(early) = tokio::time::timeout(Duration::from_millis(500), &mut second).await {
        panic!("second write finished while the first held the write limit: {early:?}");
    }
    // A second write let through would by now have synced and reached its
    // own (held) rename.
    assert_eq!(
        gate.entered(),
        1,
        "a second write reached its rename while the first was still renaming"
    );

    gate.release(2);
    join_bounded(second).await?;
    if let Some(first) = first {
        assert_eq!(join_bounded(first).await?, VALUE1.len() as u64);
    }
    // The second write got the permit only after the first write's task had
    // finished, so the first blob, even a cancelled caller's, is published.
    assert_eq!(gate.entered(), 2);
    assert_eq!(
        store.get_part_unchunked(first_digest, 0, None).await?,
        VALUE1.as_bytes()
    );
    assert_eq!(
        store.get_part_unchunked(second_digest, 0, None).await?,
        VALUE2.as_bytes()
    );
    wait_for_empty_temp_dir(&temp_path).await
}

#[cfg(target_family = "unix")]
#[nativelink_test(flavor = "multi_thread", worker_threads = 4)]
async fn write_limit_spans_rename() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new(GatedRenames::ContentPublishes, 0, 0, 2);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    check_write_limit_spans_rename(&GATE, rename_fn, false).await
}

#[cfg(target_family = "unix")]
#[nativelink_test(flavor = "multi_thread", worker_threads = 4)]
async fn write_limit_survives_cancelled_writer() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new(GatedRenames::ContentPublishes, 0, 0, 2);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    check_write_limit_spans_rename(&GATE, rename_fn, true).await
}

/// A write whose rename fails must release the write permit and remove its
/// temporary file, so later writes through both entry points still succeed.
#[cfg(target_family = "unix")]
#[nativelink_test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_rename_releases_write_limit() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new(GatedRenames::ContentPublishes, u32::MAX, 0b1, 3);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _release = ReleaseAllOnDrop(&GATE);
    let temp_path = make_temp_path("temp_path");
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: temp_path.clone(),
            max_concurrent_writes: 1,
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    let first_digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let second_digest = DigestInfo::try_new(HASH2, VALUE2.len())?;

    let err = store
        .update_oneshot(first_digest, VALUE1.into())
        .await
        .expect_err("the first rename is made to fail");
    assert!(
        format!("{err:?}").contains("Failed to rename temp file to final path"),
        "unexpected error: {err:?}"
    );
    assert!(store.has(first_digest).await?.is_none());
    wait_for_empty_temp_dir(&temp_path).await?;

    tokio::time::timeout(
        TASK_JOIN_LIMIT,
        store.update_oneshot(first_digest, VALUE1.into()),
    )
    .await
    .expect("a write after a failed rename did not get the write permit")?;
    let written = tokio::time::timeout(
        TASK_JOIN_LIMIT,
        stream_update(&store, second_digest, VALUE2),
    )
    .await
    .expect("a streamed write after a failed rename did not get the write permit")?;
    assert_eq!(written, VALUE2.len() as u64);
    assert_eq!(GATE.entered(), 3);
    assert_eq!(
        store.get_part_unchunked(first_digest, 0, None).await?,
        VALUE1.as_bytes()
    );
    assert_eq!(
        store.get_part_unchunked(second_digest, 0, None).await?,
        VALUE2.as_bytes()
    );
    wait_for_empty_temp_dir(&temp_path).await
}

/// A rename can block in the kernel for a long time (renames between two
/// directories on one filesystem are serialized). It must therefore not run
/// on a runtime thread: while one write's rename is held, other tasks on the
/// same runtime keep running. This runtime has a single thread, so a rename
/// run on it would block everything else, this test included; a
/// blocking-pool watchdog then releases the rename and the test fails.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn held_rename_does_not_block_other_tasks() -> Result<(), Error> {
    const WATCHDOG_LIMIT: Duration = Duration::from_secs(10);
    static GATE: RenameGate = RenameGate::new(GatedRenames::ContentPublishes, 0, 0, 1);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _release = ReleaseAllOnDrop(&GATE);
    let watchdog = spawn_blocking!("held_rename_watchdog", || GATE
        .release_after(WATCHDOG_LIMIT, 1));
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;

    let write = background_spawn!("held_rename_write", {
        let store = store.clone();
        async move { store.update_oneshot(digest, VALUE1.into()).await }
    });
    GATE.wait_entered(1).await?;

    // While the rename is held, an unrelated task and a timer both complete.
    let unrelated = background_spawn!("held_rename_unrelated", async { 7 });
    assert_eq!(join_bounded(unrelated).await, 7);
    sleep(Duration::from_millis(1)).await;
    assert_eq!(
        GATE.released(),
        0,
        "other tasks ran only after the watchdog released the rename: \
         the held rename blocked the runtime thread"
    );

    GATE.release(1);
    let watchdog_fired = tokio::time::timeout(TASK_JOIN_LIMIT, watchdog)
        .await
        .expect("watchdog did not finish within TASK_JOIN_LIMIT")
        .expect("watchdog panicked");
    assert!(!watchdog_fired, "the watchdog had to release the rename");
    join_bounded(write).await?;
    assert_eq!(GATE.entered(), 1);
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        VALUE1.as_bytes()
    );
    Ok(())
}

/// A remove callback that does not finish until it is released, as a real
/// callback doing filesystem work (`ExecutableVariantRemover`) may not.
/// Records whether its future was dropped before it finished.
#[derive(Debug, Default)]
struct HeldRemoveCallback {
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
    finished: AtomicBool,
    abandoned: AtomicBool,
}

impl HeldRemoveCallback {
    async fn wait_entered(&self) {
        tokio::time::timeout(TASK_JOIN_LIMIT, self.entered.notified())
            .await
            .expect("no remove callback started within TASK_JOIN_LIMIT");
    }

    async fn wait_finished(&self) {
        tokio::time::timeout(TASK_JOIN_LIMIT, async {
            while !self.finished.load(Ordering::Acquire) {
                sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the remove callback did not finish within TASK_JOIN_LIMIT");
    }
}

impl RemoveItemCallback for HeldRemoveCallback {
    fn callback<'a>(
        &'a self,
        _store_key: StoreKey<'a>,
    ) -> core::pin::Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        struct AbandonGuard<'a>(&'a HeldRemoveCallback);
        impl Drop for AbandonGuard<'_> {
            fn drop(&mut self) {
                if !self.0.finished.load(Ordering::Acquire) {
                    self.0.abandoned.store(true, Ordering::Release);
                }
            }
        }
        Box::pin(async move {
            let _guard = AbandonGuard(self);
            self.entered.notify_one();
            self.release.notified().await;
            self.finished.store(true, Ordering::Release);
        })
    }
}

/// Starts a write of `VALUE1` to `digest` that reaches the duplicate check
/// against the stored entry, waits until that check has removed the stored
/// entry from the map and is running its remove callback, then cancels the
/// writer. The removed entry must still be retired: its callback finishes
/// once released and no temporary file is left behind.
async fn cancel_upload_while_duplicate_check_removes_entry(
    store: &Arc<FilesystemStore>,
    temp_path: &str,
    digest: DigestInfo,
    callback: &HeldRemoveCallback,
) -> Result<(), Error> {
    let upload = background_spawn!("cancelled_upload_during_duplicate_check", {
        let store = store.clone();
        async move { store.update_oneshot(digest, VALUE1.into()).await }
    });
    callback.wait_entered().await;
    assert_eq!(
        store.get_evicting_map().len_for_test(),
        0,
        "the duplicate check removed the stored entry before its callback"
    );
    upload.abort();
    assert!(
        upload
            .await
            .expect_err("the upload was waiting on the callback")
            .is_cancelled()
    );
    assert!(
        !callback.abandoned.load(Ordering::Acquire),
        "cancelling the writer abandoned the cleanup of the entry it removed"
    );
    callback.release.notify_one();
    callback.wait_finished().await;
    wait_for_empty_temp_dir(temp_path).await
}

/// The duplicate check runs before the write permit is requested, and its
/// lookup reaps an expired stored entry. Cancelling the writer then must not
/// abandon that entry's remove callbacks and `unref()`: otherwise its content
/// file stays on disk, untracked by the map, and is loaded again at restart.
#[nativelink_test]
async fn cancelled_upload_still_retires_expired_entry() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        eviction_policy: Some(EvictionPolicy {
            max_seconds: 1,
            ..Default::default()
        }),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    let old_file = content_file_path(&content_path, &digest.into()).await?;
    let callback = Arc::new(HeldRemoveCallback::default());
    store.clone().register_remove_callback(callback.clone())?;
    // Expiry compares whole seconds of wall-clock time since the store was
    // created: an entry inserted in second `s` expires in second `s + 2`.
    sleep(Duration::from_millis(2100)).await;

    cancel_upload_while_duplicate_check_removes_entry(&store, &temp_path, digest, &callback)
        .await?;
    assert!(
        !Path::new(&old_file).exists(),
        "the expired content file was left on disk"
    );
    assert!(!content_file_exists(&content_path, &digest.into()).await?);
    assert_eq!(store.get_evicting_map().len_for_test(), 0);
    Ok(())
}

/// As [`cancelled_upload_still_retires_expired_entry`], for the duplicate
/// check's other removal: a stored entry whose file has disappeared from
/// disk is dropped from the map when the check fails to open it.
#[nativelink_test]
async fn cancelled_upload_still_retires_stale_entry() -> Result<(), Error> {
    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: temp_path.clone(),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    store.update_oneshot(digest, VALUE1.into()).await?;
    std::fs::remove_file(content_file_path(&content_path, &digest.into()).await?)?;
    let callback = Arc::new(HeldRemoveCallback::default());
    store.clone().register_remove_callback(callback.clone())?;

    cancel_upload_while_duplicate_check_removes_entry(&store, &temp_path, digest, &callback)
        .await?;
    assert!(!content_file_exists(&content_path, &digest.into()).await?);
    assert_eq!(store.get_evicting_map().len_for_test(), 0);
    Ok(())
}

/// An eviction retires its entries one at a time. Each `unref()` renames the
/// entry's file into the temp directory, and the kernel performs one rename
/// between two directories at a time per filesystem: unrefs run together (as
/// they did) retire nothing faster, but each held a blocking-pool thread, so
/// one eviction could take every thread from the process's other file work.
#[nativelink_test]
async fn eviction_unrefs_entries_one_at_a_time() -> Result<(), Error> {
    const EVICTED_ENTRIES: u64 = 32;
    static UNREFS_IN_PROGRESS: AtomicU64 = AtomicU64::new(0);
    static PEAK_UNREFS_IN_PROGRESS: AtomicU64 = AtomicU64::new(0);
    static UNREFS_FINISHED: AtomicU64 = AtomicU64::new(0);

    struct LocalHooks {}
    impl FileEntryHooks for LocalHooks {
        async fn on_unref_async<Fe: FileEntry>(_entry: &Fe) {
            let in_progress = UNREFS_IN_PROGRESS.fetch_add(1, Ordering::SeqCst) + 1;
            PEAK_UNREFS_IN_PROGRESS.fetch_max(in_progress, Ordering::SeqCst);
            // Waiting, as a rename waits for its blocking-pool thread, lets
            // any unref started together with this one start too.
            for _ in 0..4 {
                tokio::task::yield_now().await;
            }
            UNREFS_IN_PROGRESS.fetch_sub(1, Ordering::SeqCst);
            UNREFS_FINISHED.fetch_add(1, Ordering::SeqCst);
        }
    }

    let content_path = make_temp_path("content_path");
    let temp_path = make_temp_path("temp_path");
    let store = Box::pin(
        FilesystemStore::<TestFileEntry<LocalHooks>>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            eviction_policy: Some(EvictionPolicy {
                // The small entries (1 byte each) plus the big one (40 bytes)
                // exceed 41 bytes until every small entry is evicted.
                max_bytes: 41,
                ..Default::default()
            }),
            block_size: 1,
            ..Default::default()
        })
        .await?,
    );
    for i in 0..EVICTED_ENTRIES {
        let digest = DigestInfo::try_new(&format!("{i:064x}"), 1)?;
        store
            .update_oneshot(digest, Bytes::from_static(b"x"))
            .await?;
    }
    assert_eq!(UNREFS_FINISHED.load(Ordering::SeqCst), 0);

    let big_digest = DigestInfo::try_new(HASH1, 40)?;
    store
        .update_oneshot(big_digest, Bytes::from(vec![b'y'; 40]))
        .await?;
    // The write queued the evictions; the drain runs them, one at a time.
    store.retirement_settled_for_test().await;

    assert_eq!(UNREFS_FINISHED.load(Ordering::SeqCst), EVICTED_ENTRIES);
    assert_eq!(
        PEAK_UNREFS_IN_PROGRESS.load(Ordering::SeqCst),
        1,
        "an eviction must unref its entries one at a time",
    );
    assert!(store.has(big_digest).await?.is_some());
    // Every evicted file was renamed into the temp directory and deleted.
    check_storage_dir_empty(&temp_path).await
}

/// Builds a current-thread runtime that counts the blocking-pool threads it
/// starts (its only threads besides the caller's).
fn runtime_counting_blocking_threads(
    started: &'static AtomicU64,
) -> Result<tokio::runtime::Runtime, Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "test needs to observe the blocking pool's threads"
    )]
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .on_thread_start(move || {
            started.fetch_add(1, Ordering::SeqCst);
        })
        .build()?;
    Ok(runtime)
}

/// Makes `count` temp-file entries, as a write in progress has before its
/// file is published. Dropping one retires it: its file is deleted.
async fn make_retirable_entries(
    store: &FilesystemStore<FileEntryImpl>,
    count: u64,
) -> Result<Vec<FileEntryImpl>, Error> {
    let mut entries = Vec::new();
    for i in 0..count {
        let key = StoreKey::Digest(DigestInfo::try_new(&format!("{i:064x}"), 1)?);
        let (entry, file, _path) = store.make_temp_file(make_temp_key(&key)).await?;
        drop(file);
        entries.push(entry);
    }
    Ok(entries)
}

async fn wait_until_retired_files_deleted(
    shared_context: &nativelink_store::filesystem_store::SharedContext,
) -> Result<(), Error> {
    tokio::time::timeout(Duration::from_secs(30), async {
        while shared_context.active_drop_spawns.load(Ordering::Acquire) > 0 {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|_| make_err!(Code::DeadlineExceeded, "retired files were not deleted"))
}

/// The files of entries retired together (an eviction retires thousands)
/// are deleted one at a time from one blocking-pool thread. Nearly all are in
/// the temp directory, where the kernel performs one unlink at a time: a task
/// per file (as before) deleted them no faster, but put one blocking thread
/// per file to sleep on the directory's lock until none was left for any
/// other file operation in the process. A file already gone is counted as
/// deleted.
#[test]
fn retired_files_are_deleted_from_one_blocking_thread() -> Result<(), Error> {
    const RETIRED_FILES: u64 = 256;
    static BLOCKING_THREADS_STARTED: AtomicU64 = AtomicU64::new(0);
    let runtime = runtime_counting_blocking_threads(&BLOCKING_THREADS_STARTED)?;
    #[expect(clippy::disallowed_methods, reason = "test drives its own runtime")]
    runtime.block_on(async {
        let temp_path = make_temp_path("temp_path");
        let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: temp_path.clone(),
            ..Default::default()
        })
        .await?;
        let mut entries = make_retirable_entries(&store, RETIRED_FILES).await?;
        let shared_context = entries[0].get_shared_context_for_test();
        let gone = entries[0]
            .get_file_path_locked(|path| async move { Ok(path) })
            .await?;
        fs::remove_file(&gone).await?;

        let started_before = BLOCKING_THREADS_STARTED.load(Ordering::SeqCst);
        drop(entries);
        wait_until_retired_files_deleted(&shared_context).await?;
        let started = BLOCKING_THREADS_STARTED.load(Ordering::SeqCst) - started_before;
        assert!(
            started <= 1,
            "deleting {RETIRED_FILES} retired files started {started} blocking threads",
        );
        check_storage_dir_empty(&temp_path).await
    })
}

/// A drain that never ran (its runtime shut down before the blocking pool
/// started it) must not stop later retirements from deleting files: the next
/// one starts a new drain, which deletes the files the lost drain left too.
#[test]
fn retirement_after_a_lost_drain_deletes_the_files_it_left() -> Result<(), Error> {
    static UNUSED_THREADS_STARTED: AtomicU64 = AtomicU64::new(0);
    let first_runtime = runtime_counting_blocking_threads(&UNUSED_THREADS_STARTED)?;
    let temp_path = make_temp_path("temp_path");
    #[expect(clippy::disallowed_methods, reason = "test drives its own runtime")]
    let (store, mut entries) = first_runtime.block_on(async {
        let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: temp_path.clone(),
            ..Default::default()
        })
        .await?;
        let entries = make_retirable_entries(&store, 2).await?;
        Ok::<_, Error>((store, entries))
    })?;
    let shared_context = entries[0].get_shared_context_for_test();
    let first_handle = first_runtime.handle().clone();
    drop(first_runtime);
    {
        // The drain this starts is dropped unrun: its runtime has shut down.
        let _runtime = first_handle.enter();
        drop(entries.remove(0));
    }
    assert_eq!(shared_context.active_drop_spawns.load(Ordering::Acquire), 1);

    let second_runtime = runtime_counting_blocking_threads(&UNUSED_THREADS_STARTED)?;
    #[expect(clippy::disallowed_methods, reason = "test drives its own runtime")]
    second_runtime.block_on(async {
        drop(entries);
        wait_until_retired_files_deleted(&shared_context).await?;
        drop(store);
        check_storage_dir_empty(&temp_path).await
    })
}

/// Cleanups overlap: each write retires the entries its insertion displaced
/// after it releases its write permit, and explicit removals retire their
/// entry themselves. However many overlap, the store runs one retirement
/// operation (a rename into the temp directory or an executable-variant
/// unlink) at a time, plus its one drain deleting retired files: the kernel
/// performs those renames one at a time per filesystem, so more operations
/// in flight only hold more blocking-pool threads.
///
/// The bound is measured as the store's closures on the blocking pool at
/// once, from submission to the end of the closure, not as threads the
/// runtime started: a closure that has just finished can still be on its
/// way to the idle list when its successor is submitted, so the pool may
/// start a thread without any more closures occupying threads at once.
#[test]
fn overlapping_retirements_hold_at_most_two_blocking_threads() -> Result<(), Error> {
    static BLOCKING_THREADS_STARTED: AtomicU64 = AtomicU64::new(0);
    check_overlapping_retirements(&BLOCKING_THREADS_STARTED, false)
}

/// The same bound holds when every removed digest also has an executable
/// variant, whose cleanup waits for its digest's variant lease and then for
/// retirement admission.
#[test]
fn overlapping_variant_retirements_hold_at_most_two_blocking_threads() -> Result<(), Error> {
    static BLOCKING_THREADS_STARTED: AtomicU64 = AtomicU64::new(0);
    check_overlapping_retirements(&BLOCKING_THREADS_STARTED, true)
}

fn check_overlapping_retirements(
    blocking_threads_started: &'static AtomicU64,
    with_executable_variants: bool,
) -> Result<(), Error> {
    const RETIRING_ENTRIES: u64 = 64;
    let runtime = runtime_counting_blocking_threads(blocking_threads_started)?;
    #[expect(clippy::disallowed_methods, reason = "test drives its own runtime")]
    runtime.block_on(async {
        let content_path = make_temp_path("content_path");
        let temp_path = make_temp_path("temp_path");
        let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
            content_path: content_path.clone(),
            temp_path: temp_path.clone(),
            ..Default::default()
        })
        .await?;
        let keys = (0..RETIRING_ENTRIES)
            .map(|i| DigestInfo::try_new(&format!("{i:064x}"), 1).map(StoreKey::Digest))
            .collect::<Result<Vec<_>, _>>()?;
        for key in &keys {
            store
                .update_oneshot(key.borrow(), Bytes::from_static(b"x"))
                .await?;
        }
        let mut variants = Vec::new();
        if with_executable_variants {
            for key in &keys {
                let StoreKey::Digest(digest) = key else {
                    unreachable!("keys are digests");
                };
                variants.push(store.get_executable_hardlink_source(digest).await?);
            }
        }
        let evicting_map = store.get_evicting_map();

        let removed =
            futures::future::join_all(keys.iter().map(|key| evicting_map.remove(key))).await;
        assert!(removed.iter().all(|removed| *removed));
        // Read the temp directory without the blocking pool, so that waiting
        // starts no thread of its own.
        let retired_dir = format!("{temp_path}/{DIGEST_FOLDER_V2}");
        tokio::time::timeout(Duration::from_secs(30), async {
            while std::fs::read_dir(&retired_dir)?.next().is_some() {
                sleep(Duration::from_millis(1)).await;
            }
            Ok::<_, Error>(())
        })
        .await
        .map_err(|_| make_err!(Code::DeadlineExceeded, "retired files were not deleted"))??;
        let peak = store.peak_blocking_retirement_for_test();
        assert_eq!(
            peak,
            BlockingRetirementPeak {
                operations: 1,
                file_deletions: 1,
            },
            "{RETIRING_ENTRIES} overlapping retirements occupied more than one \
             blocking-pool closure of a kind at once (threads started: {})",
            blocking_threads_started.load(Ordering::SeqCst),
        );
        for key in &keys {
            assert!(!content_file_exists(&content_path, key).await?);
        }
        for variant in &variants {
            assert!(
                !Path::new(variant).exists(),
                "retired variant {} remains",
                variant.display()
            );
        }
        Ok(())
    })
}

/// How many of this test's log lines contain `part`.
#[cfg(target_family = "unix")]
fn count_log_lines(
    logs_assert: impl Fn(&dyn Fn(&[&str]) -> Result<(), String>),
    part: &str,
) -> usize {
    let count = core::cell::Cell::new(0);
    logs_assert(&|lines: &[&str]| {
        count.set(lines.iter().filter(|line| line.contains(part)).count());
        Ok(())
    });
    count.get()
}

/// Waits until `count` of this test's log lines contain `part`.
#[cfg(target_family = "unix")]
async fn wait_for_log_lines(
    logs_assert: impl Fn(&dyn Fn(&[&str]) -> Result<(), String>) + Copy + Send + Sync,
    part: &str,
    count: usize,
) -> Result<(), Error> {
    tokio::time::timeout(TASK_JOIN_LIMIT, async {
        while count_log_lines(logs_assert, part) < count {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|_| {
        make_err!(
            Code::DeadlineExceeded,
            "never logged {count} lines with {part}"
        )
    })
}

/// A write whose caller is dropped while it waits for a
/// `max_concurrent_writes` permit is in no detached stage: it is reported so,
/// with nothing inserted, it never reports an end, and nothing is stored. A
/// write whose caller got its result is never reported as dropped.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn write_dropped_before_its_write_permit_reports_no_detached_stage() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new(GatedRenames::ContentPublishes, 0, 0, 1);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _release = ReleaseAllOnDrop(&GATE);
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            max_concurrent_writes: 1,
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    let held = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let dropped = DigestInfo::try_new(HASH2, VALUE2.len())?;
    let held_key = StoreKey::Digest(held).as_str().into_owned();
    let dropped_key = StoreKey::Digest(dropped).as_str().into_owned();
    let logs = |check: &dyn Fn(&[&str]) -> Result<(), String>| logs_assert(check);

    // The held write keeps the only write permit inside its rename.
    let holder = background_spawn!("held_write", {
        let store = store.clone();
        async move { store.update_oneshot(held, VALUE1.into()).await }
    });
    GATE.wait_entered(1).await?;
    let waiter = background_spawn!("write_permit_waiter", {
        let store = store.clone();
        async move { store.update_oneshot(dropped, VALUE2.into()).await }
    });
    wait_for_log_lines(logs, &format!("key={dropped_key} phase=write_permit"), 1).await?;

    waiter.abort();
    assert!(waiter.await.is_err_and(|err| err.is_cancelled()));
    assert_eq!(
        count_log_lines(
            logs,
            &format!(
                "Caller stopped waiting for a filesystem store write key={dropped_key} \
                 phase=write_permit detached_stage=false inserted=false"
            )
        ),
        1
    );

    GATE.release(1);
    join_bounded(holder).await?;
    assert_eq!(
        store.has(dropped).await?,
        None,
        "the dropped write stored nothing"
    );
    assert_eq!(
        count_log_lines(
            logs,
            &format!("ended after its caller stopped waiting key={dropped_key}")
        ),
        0,
        "a write that started no detached work never ends on its own"
    );
    assert_eq!(
        count_log_lines(
            logs,
            &format!("Caller stopped waiting for a filesystem store write key={held_key}")
        ),
        0,
        "the held write's caller got its result"
    );
    Ok(())
}

/// A write whose caller is dropped while its detached finalizer is renaming
/// the file goes on: it is reported as in a detached stage with nothing
/// inserted yet, and once the rename ends it reports that it inserted its
/// file, which is then stored.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn write_dropped_during_publication_reports_its_detached_end() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new(GatedRenames::ContentPublishes, 0, 0, 1);
    fn rename_fn(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _release = ReleaseAllOnDrop(&GATE);
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            ..Default::default()
        },
        rename_fn,
    )
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let key = StoreKey::Digest(digest).as_str().into_owned();
    let logs = |check: &dyn Fn(&[&str]) -> Result<(), String>| logs_assert(check);

    let writer = background_spawn!("publishing_write", {
        let store = store.clone();
        async move { store.update_oneshot(digest, VALUE1.into()).await }
    });
    GATE.wait_entered(1).await?;
    writer.abort();
    assert!(writer.await.is_err_and(|err| err.is_cancelled()));
    assert_eq!(
        count_log_lines(
            logs,
            &format!(
                "Caller stopped waiting for a filesystem store write key={key} \
                 phase=publish detached_stage=true inserted=false"
            )
        ),
        1
    );
    let ended = format!("Filesystem store write ended after its caller stopped waiting key={key}");
    assert_eq!(count_log_lines(logs, &ended), 0, "the rename is still held");

    GATE.release(1);
    wait_for_log_lines(logs, &ended, 1).await?;
    logs_assert(|lines: &[&str]| {
        let line = lines.iter().find(|line| line.contains(&ended)).unwrap();
        if line.contains("inserted=true outcome=inserted") {
            Ok(())
        } else {
            Err(format!("unexpected end: {line}"))
        }
    });
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        VALUE1.as_bytes()
    );
    Ok(())
}

/// A write whose caller is dropped after its publication inserted its file,
/// while it retires the generation it displaced, is reported as in a
/// detached stage with its file inserted: its blob is already readable. It
/// reports its end once retirement is admitted.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn displacing_write_returns_while_retirement_is_held() -> Result<(), Error> {
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let key = StoreKey::Digest(digest).as_str().into_owned();
    let logs = |check: &dyn Fn(&[&str]) -> Result<(), String>| logs_assert(check);
    let retiring = format!("key={key} phase=retire");

    // The first generation; publishing it displaces nothing.
    store.update_oneshot(digest, VALUE1.into()).await?;
    assert_eq!(count_log_lines(logs, &retiring), 1);

    // Different content of the same size is no duplicate, so its publication
    // displaces the first generation. Its retirement is only queued: the
    // write returns although every retirement of the store is held.
    let admission = store.hold_retirement_admission_for_test().await;
    tokio::time::timeout(
        Duration::from_secs(5),
        store.update_oneshot(digest, VALUE2.into()),
    )
    .await
    .expect("the displacing write waited for the retirement admission")?;
    assert_eq!(count_log_lines(logs, &retiring), 2);
    assert_eq!(
        store.get_part_unchunked(digest, 0, None).await?,
        VALUE2.as_bytes(),
    );
    assert!(store.retirement_accounting().queued_bytes > 0);
    drop(admission);
    tokio::time::timeout(Duration::from_secs(5), store.retirement_settled_for_test())
        .await
        .expect("retirement did not finish");
    assert_eq!(store.retirement_accounting().queued_bytes, 0);
    Ok(())
}

/// Polls `write`, a write's caller, until its duplicate check has started,
/// and returns at once, before yielding: on this test's single-threaded
/// runtime the check's own task cannot have run yet. Only this test polls the
/// caller; the duplicate check runs in its own task.
#[cfg(target_family = "unix")]
async fn poll_until_duplicate_check_starts(
    logs_assert: impl Fn(&dyn Fn(&[&str]) -> Result<(), String>) + Copy + Send + Sync,
    key: &str,
    write: &mut core::pin::Pin<Box<impl Future<Output = Result<(), Error>> + Send>>,
) -> Result<(), Error> {
    let started = format!("key={key} phase=duplicate_check write_id");
    tokio::time::timeout(TASK_JOIN_LIMIT, async {
        loop {
            assert!(
                poll!(write.as_mut()).is_pending(),
                "the write returned before its duplicate check started"
            );
            if count_log_lines(logs_assert, &started) != 0 {
                return;
            }
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|_| make_err!(Code::DeadlineExceeded, "the duplicate check never started"))
}

/// A caller dropped after its write's duplicate check has finished, but
/// before the caller took up the answer, still gets the write an end: the
/// caller-drop report carries it, and nothing was inserted.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn write_dropped_with_an_untaken_duplicate_check_answer_reports_its_end() -> Result<(), Error>
{
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let key = StoreKey::Digest(digest).as_str().into_owned();
    let logs = |check: &dyn Fn(&[&str]) -> Result<(), String>| logs_assert(check);

    let mut write = Box::pin(store.update_oneshot(digest, VALUE1.into()));
    poll_until_duplicate_check_starts(logs, &key, &mut write).await?;
    // The check finishes in its own task while the caller is not polled.
    wait_for_log_lines(
        logs,
        &format!("key={key} phase=duplicate_checked write_id"),
        1,
    )
    .await?;
    drop(write);

    assert_eq!(
        count_log_lines(
            logs,
            &format!(
                "Caller stopped waiting for a filesystem store write key={key} \
                 phase=duplicate_checked detached_stage=false inserted=false"
            )
        ),
        1
    );
    assert_eq!(
        count_log_lines(
            logs,
            "outcome=Some(\"not inserted: its caller stopped waiting in duplicate_checked\")"
        ),
        1,
        "the caller-drop report must carry the write's end"
    );
    assert_eq!(store.has(digest).await?, None, "nothing was inserted");
    Ok(())
}

/// A caller dropped while its write's duplicate check is still running: the
/// check goes on in its own task, and when it finishes it ends the write,
/// as nobody will take up its answer.
#[cfg(target_family = "unix")]
#[nativelink_test]
async fn write_dropped_during_its_duplicate_check_reports_the_checks_end() -> Result<(), Error> {
    let store = FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: make_temp_path("content_path"),
        temp_path: make_temp_path("temp_path"),
        ..Default::default()
    })
    .await?;
    let digest = DigestInfo::try_new(HASH1, VALUE1.len())?;
    let key = StoreKey::Digest(digest).as_str().into_owned();
    let logs = |check: &dyn Fn(&[&str]) -> Result<(), String>| logs_assert(check);

    let mut write = Box::pin(store.update_oneshot(digest, VALUE1.into()));
    poll_until_duplicate_check_starts(logs, &key, &mut write).await?;
    assert_eq!(
        count_log_lines(logs, &format!("key={key} phase=duplicate_checked write_id")),
        0,
        "the check must still be running when the caller is dropped"
    );
    drop(write);
    assert_eq!(
        count_log_lines(
            logs,
            &format!(
                "Caller stopped waiting for a filesystem store write key={key} \
                 phase=duplicate_check detached_stage=true inserted=false"
            )
        ),
        1
    );

    let ended = format!("Filesystem store write ended after its caller stopped waiting key={key}");
    wait_for_log_lines(logs, &ended, 1).await?;
    assert_eq!(
        count_log_lines(
            logs,
            "outcome=not inserted: no caller took up the duplicate check"
        ),
        1
    );
    assert_eq!(store.has(digest).await?, None, "nothing was inserted");
    Ok(())
}

// PANIC-COV-01 (review t_f5445900): an *unref* unit whose future panics. The
// drain abandons it and goes on to the units behind it; its bytes are
// released exactly once, so a write waiting for space under the cap gets it;
// the rest of its batch and a second, later eviction batch still retire; and
// its file stays where the panic left it, in the content directory (the panic
// comes before its rename).
#[nativelink_test]
async fn a_panicking_unref_unit_is_released_once_and_the_drain_continues() -> Result<(), Error> {
    const N: u64 = 40;
    const SIZE: u64 = 1024;
    const WAIT: Duration = Duration::from_secs(10);
    static PANICKED: AtomicBool = AtomicBool::new(false);
    static PANICKED_PATH: LazyLock<Mutex<Option<OsString>>> = LazyLock::new(|| Mutex::new(None));
    static PANIC_NOW: tokio::sync::Notify = tokio::sync::Notify::const_new();
    struct PanicsOnFirstUnref;
    impl FileEntryHooks for PanicsOnFirstUnref {
        async fn on_unref_async<Fe: FileEntry>(entry: &Fe) {
            if PANICKED.swap(true, Ordering::SeqCst) {
                return;
            }
            let path = entry
                .get_file_path_locked(|path| async move { Ok(path) })
                .await
                .expect("the entry's path");
            *PANICKED_PATH.lock() = Some(path);
            // Suspended until the test lets it panic, in a resumed poll.
            PANIC_NOW.notified().await;
            panic!("test unref panics once");
        }
    }
    fn blob(index: u64) -> (DigestInfo, Bytes) {
        let mut data = format!("panicking-unref-blob-{index}-").into_bytes();
        data.resize(usize::try_from(SIZE).unwrap(), b'x');
        let hash: [u8; 32] = Sha256::digest(&data).into();
        (DigestInfo::new(hash, SIZE), Bytes::from(data))
    }
    async fn eventually(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = tokio::time::Instant::now() + WAIT;
        while !condition() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for {what}"
            );
            sleep(Duration::from_millis(5)).await;
        }
    }
    async fn content_files_of(content_path: &str, digest: DigestInfo) -> usize {
        let prefix = format!("{digest}-");
        let mut count = 0;
        let mut entries = tokio::fs::read_dir(format!("{content_path}/{DIGEST_FOLDER_V2}"))
            .await
            .expect("content directory");
        while let Some(entry) = entries.next_entry().await.expect("read content directory") {
            if entry.file_name().to_string_lossy().starts_with(&prefix) {
                count += 1;
            }
        }
        count
    }

    let content_path = make_temp_path("content_path");
    let store = FilesystemStore::<TestFileEntry<PanicsOnFirstUnref>>::new(&FilesystemSpec {
        content_path: content_path.clone(),
        temp_path: make_temp_path("temp_path"),
        eviction_policy: Some(EvictionPolicy {
            max_bytes: usize::try_from(N * SIZE).unwrap(),
            evict_bytes: usize::try_from(N * SIZE / 2).unwrap(),
            ..Default::default()
        }),
        block_size: 1,
        ..Default::default()
    })
    .await?;
    let mut digests = Vec::new();
    for index in 0..N - 1 {
        let (digest, data) = blob(index);
        store.update_oneshot(digest, data).await?;
        digests.push(digest);
    }
    tokio::time::timeout(WAIT, store.retirement_settled_for_test())
        .await
        .expect("filled store settles");
    assert!(!PANICKED.load(Ordering::SeqCst), "nothing evicted yet");

    // An eviction batch. The drain runs its units one at a time; its first
    // unref is suspended in the hook, before it asks for the retirement
    // admission, so the queue stands still with that unit's bytes still
    // counted. The test then takes the admission, so every unit behind the
    // panicking one waits for it, and lets the unit panic. The bytes still
    // queued then are those less the panicked unit's: released once, not
    // twice, not never.
    let (digest, data) = blob(N);
    store.update_oneshot(digest, data).await?;
    eventually("the first unref to start", || {
        PANICKED.load(Ordering::SeqCst)
    })
    .await;
    let admission = store.hold_retirement_admission_for_test().await;
    let queued = store.retirement_accounting();
    assert!(queued.queued_bytes > 2 * SIZE, "{queued:?}");
    assert_eq!(queued.panicked_units_total, 0, "{queued:?}");
    let batch_bytes = queued.queued_bytes;
    PANIC_NOW.notify_one();
    eventually("the first unref to panic", || {
        store.retirement_accounting().panicked_units_total == 1
    })
    .await;
    sleep(Duration::from_millis(100)).await;
    let after_panic = store.retirement_accounting();
    assert_eq!(after_panic.panicked_units_total, 1, "{after_panic:?}");
    assert_eq!(
        after_panic.queued_bytes,
        batch_bytes - SIZE,
        "the panicked unit's bytes are released exactly once: {after_panic:?}"
    );
    assert!(after_panic.draining, "{after_panic:?}");

    // A write waiting for space under the cap gets it once the rest runs.
    store.set_retirement_reservation_cap(Some(SIZE));
    let waiter = background_spawn!("waiting_write", {
        let store = store.clone();
        async move {
            let (digest, data) = blob(N + 1);
            store.update_oneshot(digest, data).await.map(|()| digest)
        }
    });
    eventually("the write to wait", || {
        store.retirement_accounting().waiting_reservations == 1
    })
    .await;
    drop(admission);
    let written = tokio::time::timeout(WAIT, waiter)
        .await
        .expect("the waiting write never got its space")
        .expect("join")?;
    store.set_retirement_reservation_cap(None);

    assert_eq!(
        store.get_part_unchunked(written, 0, None).await?,
        blob(N + 1).1
    );
    tokio::time::timeout(WAIT, store.retirement_settled_for_test())
        .await
        .expect("the first batch settles after the panic");
    let panicked_path = PANICKED_PATH.lock().clone().expect("the panicked path");
    let panicked_name = Path::new(&panicked_path)
        .file_name()
        .expect("a file name")
        .to_string_lossy()
        .into_owned();
    let is_panicked = |digest: &DigestInfo| panicked_name.starts_with(&format!("{digest}-"));

    // The first batch left about half the store resident. Enough later writes
    // fill it again past `max_bytes`, so a second batch evicts the oldest
    // residents: every original blob the first batch spared.
    let mut spared = Vec::new();
    for digest in &digests {
        if !is_panicked(digest) && content_files_of(&content_path, *digest).await == 1 {
            spared.push(*digest);
        }
    }
    assert!(!spared.is_empty(), "the first batch spared some blobs");
    for index in N + 2..=N + 22 {
        let (digest, data) = blob(index);
        store.update_oneshot(digest, data).await?;
    }
    tokio::time::timeout(WAIT, store.retirement_settled_for_test())
        .await
        .expect("retirement settles after the second batch");
    for digest in &spared {
        assert_eq!(
            content_files_of(&content_path, *digest).await,
            0,
            "the second batch retired spared blob {digest}"
        );
    }
    let settled = store.retirement_accounting();
    assert_eq!(
        (
            settled.queued_units,
            settled.queued_bytes,
            settled.waiting_reservations,
            settled.panicked_units_total
        ),
        (0, 0, 0, 1),
        "{settled:?}"
    );

    // The panicked unit's file is still in the content directory; every
    // other original blob was retired by one batch or the other.
    assert!(
        Path::new(&panicked_path).exists(),
        "the panicked unit's file stays at {panicked_path:?}"
    );
    for digest in &digests {
        assert_eq!(
            content_files_of(&content_path, *digest).await,
            usize::from(is_panicked(digest)),
            "content files left of original blob {digest}"
        );
    }
    Ok(())
}
