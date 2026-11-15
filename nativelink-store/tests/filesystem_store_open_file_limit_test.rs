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

// Copyright 2024 The NativeLink Authors. All rights reserved.

//! `FilesystemStore` writes when open-file permits run short.
//!
//! This is its own test binary because its tests reserve almost every permit
//! of the process-wide `OPEN_FILE_SEMAPHORE`, which would starve other tests
//! running beside them in the same process; [`SERIAL`] runs its own tests one
//! at a time. No file descriptors are opened to reserve the permits.

use core::time::Duration;
use std::ffi::OsStr;
use std::sync::{Condvar, Mutex};

use nativelink_config::stores::{EvictionPolicy, FilesystemSpec};
use nativelink_error::{Code, Error, make_err};
use nativelink_macro::nativelink_test;
use nativelink_store::filesystem_store::{FileEntryImpl, FilesystemStore};
use nativelink_util::background_spawn;
use nativelink_util::common::{DigestInfo, make_temp_path};
use nativelink_util::fs::OPEN_FILE_SEMAPHORE;
use nativelink_util::store_trait::StoreLike;
use pretty_assertions::assert_eq;
use tokio::time::{sleep, timeout};

const HASH_A: &str = "0123456789abcdef000000000000000000010000000000000123456789abcdef";
const HASH_B: &str = "0123456789abcdef000000000000000000020000000000000123456789abcdef";
const HASH_C: &str = "0123456789abcdef000000000000000000030000000000000123456789abcdef";
const VALUE_A: &str = "aaaaaaaaaa";
const VALUE_B: &str = "bbbbbbbbbb";
const VALUE_C: &str = "cccccccccc";

/// Bounds every wait in this test. A correct store finishes each step in
/// milliseconds.
const STEP_LIMIT: Duration = Duration::from_secs(10);

/// Each test reserves open-file permits of the whole process.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Content renames, counted from the first. The rename numbered
/// [`HELD_RENAME`] waits until [`RenameGate::release`]. Each test owns its
/// own `static` gate, because a store takes its rename hook as a plain `fn`
/// pointer.
#[derive(Debug)]
struct RenameGate {
    state: Mutex<(u32, bool)>,
    changed: Condvar,
}

/// The seeding write publishes with rename 0, the held write with rename 1.
const HELD_RENAME: u32 = 1;

impl RenameGate {
    const fn new() -> Self {
        Self {
            state: Mutex::new((0, false)),
            changed: Condvar::new(),
        }
    }

    fn rename(&self, from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        let mut state = self.state.lock().unwrap();
        let index = state.0;
        state.0 += 1;
        self.changed.notify_all();
        if index == HELD_RENAME {
            state = self
                .changed
                .wait_timeout_while(state, Duration::from_mins(1), |state| !state.1)
                .unwrap()
                .0;
        }
        drop(state);
        std::fs::rename(from, to)
    }

    fn release(&self) {
        self.state.lock().unwrap().1 = true;
        self.changed.notify_all();
    }

    async fn wait_entered(&self, count: u32) -> Result<(), Error> {
        for _ in 0..10_000 {
            if self.state.lock().unwrap().0 >= count {
                return Ok(());
            }
            sleep(Duration::from_millis(1)).await;
        }
        Err(make_err!(
            Code::DeadlineExceeded,
            "Only {} of {count} renames started",
            self.state.lock().unwrap().0
        ))
    }
}

/// Releases the held rename when dropped, so a failing assertion cannot strand
/// a blocking-pool thread (and hang runtime shutdown) inside the gate.
#[derive(Debug)]
struct ReleaseOnDrop(&'static RenameGate);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// Reserves all but `left` of the process's open-file permits.
async fn reserve_open_file_permits_leaving(left: usize) -> tokio::sync::SemaphorePermit<'static> {
    let spare = OPEN_FILE_SEMAPHORE.available_permits();
    OPEN_FILE_SEMAPHORE
        .acquire_many(u32::try_from(spare - left).expect("open-file limit fits in u32"))
        .await
        .expect("open-file semaphore is never closed")
}

fn available_open_file_permits() -> usize {
    OPEN_FILE_SEMAPHORE.available_permits()
}

/// Polls until `done` holds, within [`STEP_LIMIT`].
async fn wait_until(what: &str, mut done: impl FnMut() -> bool) -> Result<(), Error> {
    timeout(STEP_LIMIT, async {
        while !done() {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|_| make_err!(Code::DeadlineExceeded, "Timed out waiting until {what}"))
}

/// With `max_concurrent_writes: 1` and two open-file permits left, a
/// duplicate upload and a unique upload are queued behind a write whose
/// publication is held. Each queued upload holds an open temp file, and so
/// one of the two permits, while it waits. The write holding the write permit
/// must therefore never wait for an open-file permit: if the duplicate's
/// comparison against the stored blob needed one while it held the write
/// permit, the unique upload could neither get that permit nor give back its
/// file, and neither upload would ever finish. Both must finish once the held
/// write is released, without either being cancelled or a limit raised.
#[nativelink_test]
async fn queued_writers_holding_open_files_do_not_block_a_duplicate() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new();
    fn gated_rename(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _serial = SERIAL.lock().await;
    let _release = ReleaseOnDrop(&GATE);
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            max_concurrent_writes: 1,
            ..Default::default()
        },
        gated_rename,
    )
    .await?;
    let digest_a = DigestInfo::try_new(HASH_A, VALUE_A.len())?;
    let digest_b = DigestInfo::try_new(HASH_B, VALUE_B.len())?;
    let digest_c = DigestInfo::try_new(HASH_C, VALUE_C.len())?;

    // B is stored, so a second upload of B is a duplicate.
    store.update_oneshot(digest_b, VALUE_B.into()).await?;

    let reserved = reserve_open_file_permits_leaving(2).await;
    let available = available_open_file_permits;

    // A takes the write permit and holds in its rename, having closed its
    // temp file.
    let upload_a = background_spawn!("open_file_limit_upload_a", {
        let store = store.clone();
        async move { store.update_oneshot(digest_a, VALUE_A.into()).await }
    });
    GATE.wait_entered(HELD_RENAME + 1).await?;
    assert_eq!(available(), 2, "the held write keeps no file open");

    // The duplicate opens its temp file, which takes one permit. A store
    // that compares it only under the write permit queues it here; one that
    // compares it first may finish it now.
    let mut upload_b = background_spawn!("open_file_limit_upload_b", {
        let store = store.clone();
        async move { store.update_oneshot(digest_b, VALUE_B.into()).await }
    });
    wait_until("the duplicate holds its temp file or has finished", || {
        upload_b.is_finished() || available() == 1
    })
    .await?;
    // Let a queued duplicate reach the write-permit queue ahead of the unique
    // upload. A correct store passes in either order; the delay only makes
    // the order that exposes the deadlock reproducible.
    sleep(Duration::from_millis(100)).await;

    // The unique upload opens its temp file, taking the last permit if the
    // duplicate still holds one, and queues for the write permit.
    let upload_c = background_spawn!("open_file_limit_upload_c", {
        let store = store.clone();
        async move { store.update_oneshot(digest_c, VALUE_C.into()).await }
    });
    wait_until("the unique upload holds its temp file", || {
        available() == usize::from(upload_b.is_finished())
    })
    .await?;

    GATE.release();
    timeout(STEP_LIMIT, upload_a)
        .await
        .expect("the held write finishes once released")
        .expect("upload A panicked")?;
    let Ok(joined_b) = timeout(STEP_LIMIT, &mut upload_b).await else {
        let available = available();
        upload_c.abort();
        let resumed = timeout(STEP_LIMIT, &mut upload_b).await.is_ok();
        panic!(
            "the duplicate upload stalled with {available} open-file permits free while \
             the unique upload waited for the write permit; it {} once that upload was \
             cancelled",
            if resumed { "resumed" } else { "did not resume" }
        );
    };
    joined_b.expect("upload B panicked")?;
    timeout(STEP_LIMIT, upload_c)
        .await
        .expect("the unique upload finishes")
        .expect("upload C panicked")?;
    drop(reserved);

    for (digest, value) in [
        (digest_a, VALUE_A),
        (digest_b, VALUE_B),
        (digest_c, VALUE_C),
    ] {
        assert_eq!(
            store.get_part_unchunked(digest, 0, None).await?,
            value.as_bytes()
        );
    }
    Ok(())
}

/// With `max_concurrent_writes: 1` and one open-file permit left, a unique
/// upload is queued behind a write whose publication is held; it holds the
/// last permit in its open temp file. Publishing the held write evicts the
/// stored entry (`max_count: 1`), and retiring the evicted file renames it
/// into the temp directory, which takes an open-file permit. That must wait
/// only after the write permit is released: otherwise the queued upload could
/// neither get the write permit nor give back its file, and neither write
/// would ever finish.
#[nativelink_test]
async fn retiring_an_evicted_file_does_not_wait_under_the_write_permit() -> Result<(), Error> {
    static GATE: RenameGate = RenameGate::new();
    fn gated_rename(from: &OsStr, to: &OsStr) -> Result<(), std::io::Error> {
        GATE.rename(from, to)
    }
    let _serial = SERIAL.lock().await;
    let _release = ReleaseOnDrop(&GATE);
    let store = FilesystemStore::<FileEntryImpl>::new_with_timeout_and_rename_fn(
        &FilesystemSpec {
            content_path: make_temp_path("content_path"),
            temp_path: make_temp_path("temp_path"),
            max_concurrent_writes: 1,
            eviction_policy: Some(EvictionPolicy {
                max_count: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        gated_rename,
    )
    .await?;
    let digest_a = DigestInfo::try_new(HASH_A, VALUE_A.len())?;
    let digest_b = DigestInfo::try_new(HASH_B, VALUE_B.len())?;
    let digest_c = DigestInfo::try_new(HASH_C, VALUE_C.len())?;

    // B is stored; inserting A will evict it.
    store.update_oneshot(digest_b, VALUE_B.into()).await?;

    let reserved = reserve_open_file_permits_leaving(1).await;
    let available = available_open_file_permits;

    let upload_a = background_spawn!("eviction_upload_a", {
        let store = store.clone();
        async move { store.update_oneshot(digest_a, VALUE_A.into()).await }
    });
    GATE.wait_entered(HELD_RENAME + 1).await?;
    assert_eq!(available(), 1, "the held write keeps no file open");

    // C opens its temp file, taking the last permit, and queues for the write
    // permit.
    let mut upload_c = background_spawn!("eviction_upload_c", {
        let store = store.clone();
        async move { store.update_oneshot(digest_c, VALUE_C.into()).await }
    });
    wait_until("the queued upload holds its temp file", || available() == 0).await?;

    GATE.release();
    let Ok(joined_a) = timeout(STEP_LIMIT, upload_a).await else {
        upload_c.abort();
        panic!(
            "the write that evicted an entry did not finish while another upload \
             waited for the write permit holding the last open-file permit"
        );
    };
    joined_a.expect("upload A panicked")?;
    timeout(STEP_LIMIT, &mut upload_c)
        .await
        .expect("the queued upload finishes")
        .expect("upload C panicked")?;
    drop(reserved);

    // With room for one entry, C, published last, evicted A as A evicted B.
    assert_eq!(
        store.get_part_unchunked(digest_c, 0, None).await?,
        VALUE_C.as_bytes()
    );
    assert!(store.has(digest_a).await?.is_none());
    assert!(store.has(digest_b).await?.is_none());
    Ok(())
}
