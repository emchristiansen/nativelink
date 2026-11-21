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

//! The filesystem store's retirement drain: entries that leave the map through
//! an insertion or a lease release are retired off the evicting path, and the
//! optional reservation cap is checked and taken atomically.

use core::future::Future;
use core::time::Duration;
use std::sync::Arc;

use bytes::Bytes;
use nativelink_config::stores::{EvictionPolicy, FilesystemSpec};
use nativelink_error::Error;
use nativelink_macro::nativelink_test;
use nativelink_store::filesystem_store::{
    DIGEST_FOLDER_V2, FileEntry, FileEntryImpl, FilesystemStore,
};
use nativelink_util::buf_channel::make_buf_channel_pair;
use nativelink_util::common::{DigestInfo, fs, make_temp_path};
use nativelink_util::store_trait::{StoreLike, UploadSizeInfo};
use sha2::{Digest, Sha256};
use tokio::time::{sleep, timeout};

/// Serializes this binary's tests: they observe process-wide state (the
/// open-file count, runtime shutdowns), so a concurrently running test would
/// disturb them.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const SIZE: usize = 1024;
const WAIT: Duration = Duration::from_secs(10);

fn blob(index: usize) -> (DigestInfo, Bytes) {
    let mut data = format!("retirement-drain-blob-{index}-").into_bytes();
    data.resize(SIZE, b'x');
    let hash: [u8; 32] = Sha256::digest(&data).into();
    (DigestInfo::new(hash, SIZE as u64), Bytes::from(data))
}

struct Dirs {
    content: String,
    temp: String,
}

fn dirs() -> Dirs {
    Dirs {
        content: make_temp_path("content_path"),
        temp: make_temp_path("temp_path"),
    }
}

async fn open_store(
    dirs: &Dirs,
    max_bytes: usize,
    evict_bytes: usize,
) -> Result<Arc<FilesystemStore>, Error> {
    FilesystemStore::<FileEntryImpl>::new(&FilesystemSpec {
        content_path: dirs.content.clone(),
        temp_path: dirs.temp.clone(),
        eviction_policy: Some(EvictionPolicy {
            max_bytes,
            evict_bytes,
            ..Default::default()
        }),
        block_size: 1,
        ..Default::default()
    })
    .await
}

async fn write(store: &Arc<FilesystemStore>, index: usize) -> Result<DigestInfo, Error> {
    let (digest, data) = blob(index);
    store.update_oneshot(digest, data).await?;
    Ok(digest)
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

async fn settle(store: &Arc<FilesystemStore>) {
    timeout(WAIT, store.retirement_settled_for_test())
        .await
        .expect("retirement did not settle");
}

/// Files named `{digest}-{generation}` in the content directory.
async fn content_files_of(dirs: &Dirs, digest: DigestInfo) -> Vec<String> {
    let prefix = format!("{digest}-");
    let mut names = Vec::new();
    let mut entries = tokio::fs::read_dir(format!("{}/{DIGEST_FOLDER_V2}", dirs.content))
        .await
        .expect("content directory");
    while let Some(entry) = entries.next_entry().await.expect("read content directory") {
        let name = entry.file_name().into_string().expect("utf-8 file name");
        if name.starts_with(&prefix) {
            names.push(name);
        }
    }
    names
}

/// Fills a store with `count` blobs and returns their digests in insertion
/// (LRU) order.
async fn fill(store: &Arc<FilesystemStore>, count: usize) -> Result<Vec<DigestInfo>, Error> {
    let mut digests = Vec::with_capacity(count);
    for index in 0..count {
        digests.push(write(store, index).await?);
    }
    settle(store).await;
    Ok(digests)
}

/// A runtime of the test's own, which it shuts down to cancel the store's
/// drain task the way a process shutdown does.
#[expect(
    clippy::disallowed_methods,
    reason = "the test drops this runtime to cancel the store's background tasks"
)]
fn own_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime")
}

#[expect(
    clippy::disallowed_methods,
    reason = "drives the test's own runtime from the test thread"
)]
fn run_on<T>(runtime: &tokio::runtime::Runtime, future: impl Future<Output = T>) -> T {
    runtime.block_on(future)
}

// Test 1 (with the window-A shape): a write whose insertion evicts a batch
// returns, and so do 32 concurrent writes behind it, while every rename of the
// store is held. Before the drain, the evicting writer retired the batch
// itself and the others queued behind it for the retirement admission.
#[nativelink_test]
async fn writes_return_while_their_eviction_batch_is_held() -> Result<(), Error> {
    const N: usize = 200;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;

    let admission = store.hold_retirement_admission_for_test().await;
    timeout(Duration::from_secs(5), write(&store, N))
        .await
        .expect("an evicting write waited for its eviction batch")?;
    let batch = store.retirement_accounting();
    assert!(
        batch.queued_bytes >= (N / 2 - 1) as u64 * SIZE as u64,
        "the batch is queued: {batch:?}"
    );
    assert_eq!(
        store.has(digests[0]).await?,
        None,
        "victims leave the map at once"
    );

    let writers: Vec<_> = (0..32)
        .map(|offset| {
            let store = store.clone();
            tokio::spawn(async move { write(&store, N + 1 + offset).await })
        })
        .collect();
    timeout(
        Duration::from_secs(5),
        futures::future::try_join_all(writers),
    )
    .await
    .expect("concurrent writes waited for a held eviction batch")
    .expect("join")
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;

    drop(admission);
    settle(&store).await;
    let settled = store.retirement_accounting();
    assert_eq!((settled.queued_units, settled.queued_bytes), (0, 0));
    assert!(settled.peak_bytes >= batch.queued_bytes);
    for victim in &digests[..N / 2 - 1] {
        assert_eq!(store.has(*victim).await?, None);
    }
    // Every victim's file left the content directory (renamed, then deleted).
    for victim in &digests[..N / 2 - 1] {
        assert_eq!(content_files_of(&dirs, *victim).await, Vec::<String>::new());
    }
    Ok(())
}

// A drain that runs longer than its report interval reports the backlog in
// the journal while it runs, and again when it has caught up; one that keeps
// up within the interval logs neither at info level. Production exports no
// metrics by default, so this is where a drain that has fallen behind shows.
// The report must come while the drain is stalled on a unit, not only once
// the unit has finished: here the first unit (the batch's first remove
// callback or unref) waits for the retirement admission, and the report is
// required before the admission is released.
#[nativelink_test]
async fn a_long_drain_reports_its_backlog() -> Result<(), Error> {
    const N: usize = 40;
    const INTERVAL: Duration = Duration::from_millis(100);
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    fill(&store, N - 1).await?;

    // Within the default interval: the batch retires without a report.
    write(&store, N).await?;
    settle(&store).await;
    logs_assert(|lines: &[&str]| match info_lines(lines) {
        0 => Ok(()),
        reports => Err(format!("{reports} info-level retirement reports")),
    });

    // A drain stalled on a unit past a nonzero interval reports while the
    // unit is still pending.
    store.set_retirement_report_interval_for_test(INTERVAL);
    let admission = store.hold_retirement_admission_for_test().await;
    for index in N + 1..2 * N {
        write(&store, index).await?;
    }
    // The drain takes a whole batch of unrefs off the queue at once, so
    // what shows that it is owed is its bytes.
    let held = store.retirement_accounting();
    assert!(held.queued_bytes > 0, "a batch is owed: {held:?}");
    let store_field = format!("store={}", dirs.content);
    let stalled_report = |lines: &[&str]| {
        lines.iter().any(|line| {
            line.contains(" INFO ")
                && line.contains("Filesystem store retirement backlog")
                && (line.contains("pending_unit=callback") || line.contains("pending_unit=unref"))
                && line.contains("queued_units=")
                && line.contains(&store_field)
        })
    };
    timeout(WAIT, async {
        while !(logs_contain("pending_unit=callback") || logs_contain("pending_unit=unref")) {
            sleep(INTERVAL / 4).await;
        }
    })
    .await
    .expect("no backlog report while the drain was stalled on a unit");
    logs_assert(|lines: &[&str]| {
        if stalled_report(lines) {
            Ok(())
        } else {
            Err("the stalled-unit report lacks its level or fields".to_string())
        }
    });
    // Still stalled: reporting neither ran nor dropped the pending unit.
    let still_held = store.retirement_accounting();
    assert!(
        still_held.draining,
        "the drain is still running: {still_held:?}"
    );
    assert!(
        still_held.queued_bytes > 0,
        "the pending unit's bytes are still queued: {still_held:?}"
    );

    drop(admission);
    settle(&store).await;
    let settled = store.retirement_accounting();
    assert_eq!((settled.queued_units, settled.queued_bytes), (0, 0));
    logs_assert(|lines: &[&str]| {
        let caught_up = lines.iter().any(|line| {
            line.contains(" INFO ") && line.contains("Filesystem store retirement drain caught up")
        });
        if caught_up {
            Ok(())
        } else {
            Err("no caught-up report".to_string())
        }
    });
    Ok(())
}

/// Info-level retirement reports among `lines`.
fn info_lines(lines: &[&str]) -> usize {
    lines
        .iter()
        .filter(|line| {
            line.contains(" INFO ")
                && (line.contains("Filesystem store retirement backlog")
                    || line.contains("Filesystem store retirement drain caught up"))
        })
        .count()
}

// Test 3: simultaneous writes cannot all pass the cap check. With a cap of 8
// blobs, 32 writes that each hold their reservation (their data is not yet
// complete) are admitted 8 at a time, and the waiters hold no open file.
#[nativelink_test]
async fn reservation_cap_admits_simultaneous_writes_atomically() -> Result<(), Error> {
    const CAP_BLOBS: usize = 8;
    const WRITERS: usize = 32;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, 1 << 30, 0).await?;
    let cap = u64::try_from(CAP_BLOBS * SIZE).expect("cap fits");
    store.set_retirement_reservation_cap(Some(cap));
    let open_before = fs::get_open_files_for_test();

    let mut senders = Vec::new();
    let mut writes = Vec::new();
    for index in 0..WRITERS {
        let (digest, data) = blob(index);
        let (mut tx, rx) = make_buf_channel_pair();
        tx.send(data.slice(..1)).await?;
        let store = store.clone();
        writes.push(tokio::spawn(async move {
            store
                .update(digest, rx, UploadSizeInfo::ExactSize(SIZE as u64))
                .await
        }));
        senders.push((tx, data));
    }

    eventually("8 reservations", || {
        store.retirement_accounting().reserved_bytes == cap
    })
    .await;
    sleep(Duration::from_millis(200)).await;
    let held = store.retirement_accounting();
    assert_eq!(held.reserved_bytes, cap, "{held:?}");
    assert!(held.peak_bytes <= cap, "{held:?}");
    assert!(
        fs::get_open_files_for_test() - open_before <= CAP_BLOBS,
        "waiting writers hold no open file"
    );

    let finishers: Vec<_> = senders
        .into_iter()
        .map(|(mut tx, data)| {
            tokio::spawn(async move {
                tx.send(data.slice(1..)).await?;
                tx.send_eof()
            })
        })
        .collect();
    for write in timeout(WAIT, futures::future::join_all(writes))
        .await
        .expect("writes")
    {
        write.expect("join")?;
    }
    for finisher in finishers {
        finisher.await.expect("join")?;
    }
    let done = store.retirement_accounting();
    assert_eq!(done.reserved_bytes, 0);
    assert!(done.peak_bytes <= cap, "{done:?}");
    for index in 0..WRITERS {
        assert_eq!(store.has(blob(index).0).await?, Some(SIZE as u64));
    }
    Ok(())
}

// Test 4: a drain dropped while running a unit (here: its runtime shuts down
// while the unit waits for the retirement admission) puts the unit back, and
// the next enqueue retires everything.
#[test]
fn drain_dropped_mid_unit_puts_it_back() -> Result<(), Error> {
    const N: usize = 40;
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let first = own_runtime();
    let (store, admission, running, digests) = run_on(&first, async {
        let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
        let digests = fill(&store, N - 1).await?;
        let admission = store.hold_retirement_admission_for_test().await;
        timeout(Duration::from_secs(5), write(&store, N))
            .await
            .expect("an evicting write waited for its eviction batch")?;
        // Wait for the drain to take its first unit and block on the
        // admission.
        let mut previous = store.retirement_accounting();
        loop {
            sleep(Duration::from_millis(100)).await;
            let now = store.retirement_accounting();
            if now == previous && now.draining {
                break;
            }
            previous = now;
        }
        Ok::<_, Error>((store, admission, previous, digests))
    })?;
    drop(first);

    let after = store.retirement_accounting();
    assert!(!after.draining, "{after:?}");
    // The running unit was a batch of unrefs, taken off the queue together:
    // all of them are back, and every owed byte is a queued unref's.
    assert!(
        after.queued_units > running.queued_units,
        "the running units were put back: {running:?} then {after:?}"
    );
    assert_eq!(after.queued_bytes, after.queued_units as u64 * SIZE as u64);
    assert_eq!(after.queued_bytes, running.queued_bytes);

    drop(admission);
    let second = own_runtime();
    run_on(&second, async {
        store.release_digests(&[]).await;
        settle(&store).await;
        let settled = store.retirement_accounting();
        assert_eq!((settled.queued_units, settled.queued_bytes), (0, 0));
        for victim in &digests[..N / 2 - 1] {
            assert_eq!(content_files_of(&dirs, *victim).await, Vec::<String>::new());
        }
        Ok(())
    })
}

// Test 5: a victim evicted, uploaded again and only then retired keeps its new
// generation: each generation has its own path.
#[nativelink_test]
async fn late_retirement_spares_a_new_generation() -> Result<(), Error> {
    const N: usize = 40;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    let admission = store.hold_retirement_admission_for_test().await;
    timeout(Duration::from_secs(5), write(&store, N))
        .await
        .expect("an evicting write waited for its eviction batch")?;
    assert_eq!(store.has(digests[0]).await?, None);
    timeout(Duration::from_secs(5), write(&store, 0))
        .await
        .expect("a write waited for a held eviction batch")?;
    assert_eq!(content_files_of(&dirs, digests[0]).await.len(), 2);

    drop(admission);
    settle(&store).await;
    let (_, data) = blob(0);
    assert_eq!(store.get_part_unchunked(digests[0], 0, None).await?, data);
    assert_eq!(content_files_of(&dirs, digests[0]).await.len(), 1);
    Ok(())
}

// Test 6: the drain does not wait behind a reader. A reader holding a victim's
// path lock gets that victim's unref handed to a detached task, and the rest of
// the batch retires meanwhile.
#[nativelink_test]
async fn drain_passes_a_victim_whose_reader_holds_its_path() -> Result<(), Error> {
    const N: usize = 40;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    // Take the entry before the other blobs are written, so the lookup does
    // not make it the most recently used one.
    let first = write(&store, 0).await?;
    let held = store.get_file_entry_for_digest(&first).await?;
    let mut digests = vec![first];
    for index in 1..N - 1 {
        digests.push(write(&store, index).await?);
    }
    settle(&store).await;
    let reader = held.get_encoded_file_path().read().await;

    timeout(Duration::from_secs(5), write(&store, N))
        .await
        .expect("an evicting write waited behind a reader of its victim")?;
    eventually("the rest of the batch", || {
        let now = store.retirement_accounting();
        now.queued_units == 0 && now.queued_bytes == 0 && !now.draining
    })
    .await;
    let passed = store.retirement_accounting();
    assert_eq!(passed.detached_in_flight, 1, "{passed:?}");
    for victim in &digests[1..N / 2 - 1] {
        assert_eq!(content_files_of(&dirs, *victim).await, Vec::<String>::new());
    }
    assert_eq!(content_files_of(&dirs, digests[0]).await.len(), 1);

    drop(reader);
    settle(&store).await;
    assert_eq!(store.retirement_accounting().detached_total, 1);
    assert_eq!(
        content_files_of(&dirs, digests[0]).await,
        Vec::<String>::new()
    );
    drop(held);
    Ok(())
}

// Test 9: releasing action-input leases does not wait for the batch it
// evicts, and that batch is larger than `evict_bytes` plus one entry: the map
// was over `max_bytes` while every entry was leased.
#[nativelink_test]
async fn lease_release_queues_an_over_limit_batch() -> Result<(), Error> {
    const LEASED: usize = 20;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, 10 * SIZE, 5 * SIZE).await?;
    let digests: Vec<_> = (0..LEASED).map(|index| blob(index).0).collect();
    for digest in &digests {
        store.lease_digest(digest);
    }
    for index in 0..LEASED {
        write(&store, index).await?;
    }
    settle(&store).await;
    assert_eq!(store.retirement_accounting().queued_bytes, 0);
    for digest in &digests {
        assert_eq!(
            store.has(*digest).await?,
            Some(SIZE as u64),
            "leased entries stay"
        );
    }

    let admission = store.hold_retirement_admission_for_test().await;
    timeout(Duration::from_secs(5), store.release_digests(&digests))
        .await
        .expect("a lease release waited for the batch it evicted");
    let batch = store.retirement_accounting();
    // Eviction continues while the map holds at least `max_bytes -
    // evict_bytes` (5 blobs), so 4 blobs stay: 16 leave in one batch, against
    // the 6 (`evict_bytes` plus one entry) that one insertion evicts.
    assert_eq!(
        batch.queued_bytes,
        (LEASED - 4) as u64 * SIZE as u64,
        "evicted below max_bytes - evict_bytes: {batch:?}"
    );
    drop(admission);
    settle(&store).await;
    assert_eq!(store.retirement_accounting().queued_bytes, 0);
    Ok(())
}

// Test 8: queued retirement that never ran (the process stopped) leaves only
// recoverable state: a restarted store indexes the victims' files again, a
// stale older generation of a resident digest is replaced and removed, and the
// newest generation is served.
#[test]
fn restart_after_unretired_queue_recovers() -> Result<(), Error> {
    const N: usize = 40;
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let first = own_runtime();
    let digests = run_on(&first, async {
        let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
        let digests = fill(&store, N - 1).await?;
        let admission = store.hold_retirement_admission_for_test().await;
        timeout(Duration::from_secs(5), write(&store, N))
            .await
            .expect("an evicting write waited for its eviction batch")?;
        assert!(store.retirement_accounting().queued_units > 0);
        // A stale lower generation of a resident digest, as a crash between
        // a supersession and its retirement leaves.
        let resident = digests[N - 2];
        let names = content_files_of(&dirs, resident).await;
        assert_eq!(names.len(), 1);
        tokio::fs::copy(
            format!("{}/{DIGEST_FOLDER_V2}/{}", dirs.content, names[0]),
            format!("{}/{DIGEST_FOLDER_V2}/{resident}-0", dirs.content),
        )
        .await
        .expect("plant stale generation");
        drop(admission);
        Ok::<_, Error>(digests)
    })?;
    drop(first);

    let second = own_runtime();
    run_on(&second, async {
        let store = open_store(&dirs, 4 * N * SIZE, N * SIZE / 2).await?;
        settle(&store).await;
        assert_eq!(store.retirement_accounting().queued_bytes, 0);
        let resident = digests[N - 2];
        let mut stale_gone = false;
        for _ in 0..200 {
            if content_files_of(&dirs, resident).await.len() == 1 {
                stale_gone = true;
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
        assert!(stale_gone, "the stale generation was removed");
        assert_ne!(
            content_files_of(&dirs, resident).await,
            vec![format!("{resident}-0")]
        );
        let (_, data) = blob(N - 2);
        assert_eq!(store.get_part_unchunked(resident, 0, None).await?, data);
        // Victims that were never retired are indexed again.
        let (_, data0) = blob(0);
        assert_eq!(store.get_part_unchunked(digests[0], 0, None).await?, data0);
        Ok(())
    })
}

// ---- Cancellation of retirement work that survives in the store (review
// t_ac102b23: DRAIN-CANCEL-01 and DRAIN-CANCEL-02). ----

#[expect(
    clippy::disallowed_methods,
    reason = "a single-threaded runtime of the test's own: its block_on returns before polling tasks it spawned"
)]
fn own_single_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

/// Three leased blobs in a store whose map holds two: releasing the first
/// lease through the map, without the store, evicts that blob and hands its
/// entry to the test instead of the retirement queue. Returns the store, the
/// blob's digest and its entry, now out of the map with its content file.
async fn store_with_evicted_entry(
    dirs: &Dirs,
) -> Result<(Arc<FilesystemStore>, DigestInfo, Arc<FileEntryImpl>), Error> {
    let store = open_store(dirs, 2 * SIZE, 0).await?;
    let digests: Vec<_> = (0..3).map(|index| blob(index).0).collect();
    for digest in &digests {
        store.lease_digest(digest);
    }
    for index in 0..3 {
        write(&store, index).await?;
    }
    let (callbacks, mut entries) = store
        .get_evicting_map()
        .release_keys_deferring_cleanup([nativelink_util::store_trait::StoreKey::Digest(
            digests[0],
        )])
        .into_parts();
    drop(callbacks);
    assert_eq!(entries.len(), 1, "exactly the released blob is evicted");
    assert_eq!(store.has(digests[0]).await?, None);
    assert_eq!(content_files_of(dirs, digests[0]).await.len(), 1);
    Ok((store, digests[0], entries.pop().expect("one entry")))
}

/// Reads the whole file through the entry's path, as a later reader holding
/// the entry does.
async fn read_through_entry(entry: &FileEntryImpl) -> Result<Vec<u8>, Error> {
    use tokio::io::AsyncReadExt;
    let mut file = entry.read_file_part(0, SIZE as u64).await?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).await?;
    Ok(data)
}

async fn temp_files(dirs: &Dirs) -> Vec<String> {
    let mut names = Vec::new();
    let mut entries = tokio::fs::read_dir(format!("{}/{DIGEST_FOLDER_V2}", dirs.temp))
        .await
        .expect("temp directory");
    while let Some(entry) = entries.next_entry().await.expect("read temp directory") {
        names.push(entry.file_name().into_string().expect("utf-8 file name"));
    }
    names
}

/// Restarts the drain with an empty enqueue, waits for it, and checks that the
/// entry was retired: its content file is gone, it still reads its bytes by
/// path, exactly one temp file holds them, and dropping it deletes that file.
async fn restart_and_check_retired(
    store: &Arc<FilesystemStore>,
    dirs: &Dirs,
    digest: DigestInfo,
    entry: Arc<FileEntryImpl>,
) -> Result<(), Error> {
    store.release_digests(&[]).await;
    settle(store).await;
    let settled = store.retirement_accounting();
    assert_eq!(
        (
            settled.queued_units,
            settled.queued_bytes,
            settled.detached_in_flight
        ),
        (0, 0, 0),
        "{settled:?}"
    );
    assert!(content_files_of(dirs, digest).await.is_empty(), "renamed");
    assert_eq!(read_through_entry(&entry).await?, blob(0).1, "read by path");
    assert_eq!(temp_files(dirs).await.len(), 1, "one renamed file");
    drop(entry);
    eventually_async("the renamed file's deletion", || async {
        temp_files(dirs).await.is_empty()
    })
    .await;
    Ok(())
}

async fn eventually_async<F: Future<Output = bool>>(what: &str, mut condition: impl FnMut() -> F) {
    let deadline = tokio::time::Instant::now() + WAIT;
    while !condition().await {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        sleep(Duration::from_millis(5)).await;
    }
}

// DRAIN-CANCEL-02 (a): a detached unref whose task is dropped before its first
// poll is put back on the queue with its bytes; the count does not leak, and
// the next enqueue retires the entry.
#[test]
fn detached_unref_dropped_before_first_poll_is_requeued() -> Result<(), Error> {
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let home = own_runtime();
    let (store, digest, entry) = run_on(&home, store_with_evicted_entry(&dirs))?;

    let doomed = own_single_thread_runtime();
    // `block_on` returns as soon as this future is done, before the runtime
    // polls the task it spawned.
    run_on(&doomed, async {
        store.detach_unref_for_test(entry.clone());
    });
    let detached = store.retirement_accounting();
    assert_eq!(
        (detached.detached_in_flight, detached.detached_started_total),
        (1, 0),
        "{detached:?}"
    );
    drop(doomed);

    let after = store.retirement_accounting();
    assert_eq!(after.detached_in_flight, 0, "no leaked count: {after:?}");
    assert_eq!(after.detached_started_total, 0, "never polled: {after:?}");
    assert_eq!(after.requeued_detached_total, 1, "{after:?}");
    assert_eq!(
        (after.queued_units, after.queued_bytes),
        (1, SIZE as u64),
        "the unref is owed again: {after:?}"
    );
    run_on(
        &home,
        restart_and_check_retired(&store, &dirs, digest, entry),
    )
}

// DRAIN-CANCEL-02 (b): a detached unref cancelled while it waits for the reader
// holding the entry's path is put back on the queue; once the reader is done,
// the next enqueue retires the entry.
#[test]
fn detached_unref_cancelled_while_waiting_for_reader_is_requeued() -> Result<(), Error> {
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let home = own_runtime();
    let (store, digest, entry) = run_on(&home, store_with_evicted_entry(&dirs))?;
    let reader = run_on(&home, entry.get_encoded_file_path().read());

    let doomed = own_runtime();
    run_on(&doomed, async {
        store.detach_unref_for_test(entry.clone());
        eventually("the detached unref to start", || {
            store.retirement_accounting().detached_started_total == 1
        })
        .await;
        // It is now waiting for the reader.
        sleep(Duration::from_millis(50)).await;
    });
    assert_eq!(store.retirement_accounting().detached_in_flight, 1);
    drop(doomed);

    let after = store.retirement_accounting();
    assert_eq!(after.detached_in_flight, 0, "no leaked count: {after:?}");
    assert_eq!(after.requeued_detached_total, 1, "{after:?}");
    assert_eq!(
        (after.queued_units, after.queued_bytes),
        (1, SIZE as u64),
        "{after:?}"
    );
    assert_eq!(
        run_on(&home, content_files_of(&dirs, digest)).len(),
        1,
        "not renamed while the reader held it"
    );
    drop(reader);
    run_on(
        &home,
        restart_and_check_retired(&store, &dirs, digest, entry),
    )
}

// DRAIN-CANCEL-01: the drain is cancelled after its rename of an entry's file
// has succeeded and before the entry has recorded the new path. The entry,
// still held, reads its bytes by path afterwards, the replayed unit does not
// rename again, and dropping the entry deletes the file the rename made.
#[test]
fn drain_cancelled_after_successful_rename_keeps_the_actual_destination() -> Result<(), Error> {
    use core::sync::atomic::{AtomicBool, Ordering};
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let home = own_runtime();
    let store = run_on(&home, open_store(&dirs, 2 * SIZE, 0))?;
    let digests: Vec<_> = (0..3).map(|index| blob(index).0).collect();
    let entry = run_on(&home, async {
        for digest in &digests {
            store.lease_digest(digest);
        }
        for index in 0..3 {
            write(&store, index).await?;
        }
        store.get_file_entry_for_digest(&digests[0]).await
    })?;

    let entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    store.set_after_retire_rename_hook_for_test(Some(Arc::new({
        let (entered, release) = (entered.clone(), release.clone());
        move || {
            if !entered.swap(true, Ordering::SeqCst) {
                while !release.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    })));

    let doomed = own_runtime();
    run_on(&doomed, async {
        // Releases the first blob's lease; the drain, on this runtime, takes
        // its unit and renames the file, then stops in the hook.
        store.release_digests(&digests[..1]).await;
        eventually("the rename", || entered.load(Ordering::SeqCst)).await;
    });
    assert!(
        run_on(&home, content_files_of(&dirs, digests[0])).is_empty(),
        "the rename has happened"
    );
    // Cancel the drain while its rename has not recorded the new path.
    doomed.shutdown_background();
    // The shutdown drops the drain's task in the background.
    let deadline = std::time::Instant::now() + WAIT;
    while store.retirement_accounting().draining {
        assert!(
            std::time::Instant::now() < deadline,
            "the drain was not dropped"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let cancelled = store.retirement_accounting();
    assert!(!cancelled.draining, "{cancelled:?}");
    assert_eq!(
        cancelled.queued_units, 1,
        "the unit was put back: {cancelled:?}"
    );
    release.store(true, Ordering::SeqCst);
    store.set_after_retire_rename_hook_for_test(None);

    run_on(&home, async {
        // The blocking rename finishes and records the path it chose.
        assert_eq!(read_through_entry(&entry).await?, blob(0).1);
        let renamed = temp_files(&dirs).await;
        assert_eq!(renamed.len(), 1, "{renamed:?}");
        restart_and_check_retired(&store, &dirs, digests[0], entry).await
    })
}

// Optional cap: a write waiting for space that is held by units a dropped
// drain left restarts the drain instead of waiting for another producer.
#[test]
fn reservation_waiter_restarts_a_dropped_drain() -> Result<(), Error> {
    const N: usize = 40;
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let home = own_runtime();
    let store = run_on(&home, async {
        let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
        fill(&store, N - 1).await?;
        Ok::<_, Error>(store)
    })?;
    let doomed = own_runtime();
    let admission = run_on(&doomed, async {
        let admission = store.hold_retirement_admission_for_test().await;
        timeout(Duration::from_secs(5), write(&store, N))
            .await
            .expect("an evicting write waited for its eviction batch")?;
        Ok::<_, Error>(admission)
    })?;
    drop(doomed);
    drop(admission);
    let left = store.retirement_accounting();
    assert!(
        !left.draining && left.queued_bytes > SIZE as u64,
        "{left:?}"
    );
    store.set_retirement_reservation_cap(Some(SIZE as u64));
    run_on(&home, async {
        timeout(WAIT, write(&store, N + 1))
            .await
            .expect("a waiting write did not restart the drain")?;
        settle(&store).await;
        assert_eq!(store.retirement_accounting().queued_bytes, 0);
        Ok(())
    })
}

// DRAIN-RESERVE-03 (review t_16dd3b0d): with a cap, a write already asleep
// waiting for space a live drain holds, on a different runtime, is woken when
// that drain's runtime shuts down, and restarts the drain itself; nothing else
// enqueues or frees space.
#[test]
fn reservation_asleep_before_drain_cancellation_restarts_it() -> Result<(), Error> {
    const N: usize = 40;
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let home = own_runtime();
    let (store, digests) = run_on(&home, async {
        let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
        let digests = fill(&store, N - 1).await?;
        Ok::<_, Error>((store, digests))
    })?;

    // A live drain on its own runtime holds a batch, stopped at the
    // retirement admission.
    let doomed = own_runtime();
    let admission = run_on(&doomed, async {
        let admission = store.hold_retirement_admission_for_test().await;
        timeout(Duration::from_secs(5), write(&store, N))
            .await
            .expect("an evicting write waited for its eviction batch")?;
        Ok::<_, Error>(admission)
    })?;
    let held = store.retirement_accounting();
    assert!(held.draining && held.queued_bytes > SIZE as u64, "{held:?}");
    store.set_retirement_reservation_cap(Some(SIZE as u64));

    // A write on the home runtime waits for space while that drain runs.
    let mut waiter = None;
    run_on(&home, async {
        let store = store.clone();
        waiter = Some(nativelink_util::background_spawn!(
            "waiting_write",
            async move { write(&store, N + 1).await }
        ));
    });
    let waiter = waiter.expect("spawned");
    run_on(&home, async {
        eventually("the write to wait", || {
            store.retirement_accounting().waiting_reservations == 1
        })
        .await;
    });
    assert!(store.retirement_accounting().draining);

    // The drain's runtime shuts down; only then may retirement proceed.
    drop(doomed);
    drop(admission);

    run_on(&home, async {
        let written = timeout(WAIT, waiter)
            .await
            .expect("the waiting write was not woken to restart the drain")
            .expect("join")?;
        settle(&store).await;
        let settled = store.retirement_accounting();
        assert_eq!(
            (
                settled.queued_units,
                settled.queued_bytes,
                settled.waiting_reservations
            ),
            (0, 0, 0),
            "{settled:?}"
        );
        // The batch was retired and the waiting write's blob reads back.
        for victim in &digests[..N / 2 - 1] {
            assert_eq!(content_files_of(&dirs, *victim).await, Vec::<String>::new());
        }
        assert_eq!(
            store.get_part_unchunked(written, 0, None).await?,
            blob(N + 1).1
        );
        Ok(())
    })
}

// Holder-diagnostic port (nix-config 530c24d7, without its timer): while the
// drain waits for the retirement admission, its backlog report names the
// operation holding the admission, here an explicit removal whose rename is
// held, before that holder is released.
#[nativelink_test]
async fn a_stalled_drain_names_the_admission_holder() -> Result<(), Error> {
    use nativelink_util::store_trait::StoreKey;
    const N: usize = 40;
    const INTERVAL: Duration = Duration::from_millis(100);
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    store.set_retirement_report_interval_for_test(INTERVAL);

    // The removal's rename holds the admission inside its blocking closure
    // until released; only the first rename is held.
    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let entered_tx = std::sync::Mutex::new(Some(entered_tx));
    let release_rx = std::sync::Mutex::new(Some(release_rx));
    store.set_after_retire_rename_hook_for_test(Some(Arc::new(move || {
        let entered = entered_tx.lock().unwrap().take();
        if let Some(tx) = entered {
            tx.send(()).unwrap();
            let release = release_rx.lock().unwrap().take();
            if let Some(rx) = release {
                rx.recv().unwrap();
            }
        }
    })));
    let removed = digests[N - 2];
    let holder = tokio::spawn({
        let map = store.get_evicting_map();
        async move { map.remove(&StoreKey::Digest(removed)).await }
    });
    tokio::task::spawn_blocking(move || entered_rx.recv_timeout(WAIT))
        .await
        .expect("join")
        .expect("the removal's rename reached its hook");

    // Insertions past the limit (the removal freed one entry) queue an
    // eviction batch; its drain waits for the admission the removal holds.
    for index in N..N + 3 {
        write(&store, index).await?;
    }
    // The drain takes a whole batch of unrefs off the queue at once, so
    // what shows that it is owed is its bytes.
    let held = store.retirement_accounting();
    assert!(held.queued_bytes > 0, "a batch is owed: {held:?}");
    timeout(WAIT, async {
        while !logs_contain("holder=Some(") {
            sleep(INTERVAL / 4).await;
        }
    })
    .await
    .expect("no backlog report named the admission holder while it held it");
    let store_field = format!("store={}", dirs.content);
    logs_assert(|lines: &[&str]| {
        let named = lines.iter().any(|line| {
            line.contains(" INFO ")
                && line.contains("Filesystem store retirement backlog")
                && line.contains(&store_field)
                && (line.contains("pending_unit=callback") || line.contains("pending_unit=unref"))
                && line.contains("holder=Some(")
                && line.contains("Rename")
        });
        if named {
            Ok(())
        } else {
            Err("no pending-unit report naming a Rename holder".to_string())
        }
    });
    assert!(
        !holder.is_finished(),
        "the holder is still running when named"
    );

    release_tx.send(()).unwrap();
    assert!(
        timeout(WAIT, holder)
            .await
            .expect("holder finished")
            .expect("join")
    );
    store.set_after_retire_rename_hook_for_test(None);
    settle(&store).await;
    let settled = store.retirement_accounting();
    assert_eq!((settled.queued_units, settled.queued_bytes), (0, 0));
    Ok(())
}

/// A remove callback whose first run waits until `release` gets a permit;
/// later runs return at once.
#[derive(Debug)]
struct BlocksFirstCall {
    called: core::sync::atomic::AtomicBool,
    release: tokio::sync::Semaphore,
}

impl Default for BlocksFirstCall {
    fn default() -> Self {
        Self {
            called: core::sync::atomic::AtomicBool::new(false),
            release: tokio::sync::Semaphore::new(0),
        }
    }
}

impl nativelink_util::store_trait::RemoveItemCallback for BlocksFirstCall {
    fn callback<'a>(
        &'a self,
        _store_key: nativelink_util::store_trait::StoreKey<'a>,
    ) -> core::pin::Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            if !self.called.swap(true, core::sync::atomic::Ordering::SeqCst) {
                drop(self.release.acquire().await.expect("release"));
            }
        })
    }
}

/// A remove callback whose first run panics; every later run counts itself.
#[derive(Debug, Default)]
struct PanicsOnce {
    panicked: core::sync::atomic::AtomicBool,
    finished: core::sync::atomic::AtomicUsize,
}

impl nativelink_util::store_trait::RemoveItemCallback for PanicsOnce {
    fn callback<'a>(
        &'a self,
        _store_key: nativelink_util::store_trait::StoreKey<'a>,
    ) -> core::pin::Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        use core::sync::atomic::Ordering;
        Box::pin(async move {
            // Suspend once first, so the panic happens in a resumed poll.
            tokio::task::yield_now().await;
            assert!(
                self.panicked.swap(true, Ordering::SeqCst),
                "test remove callback panics once"
            );
            self.finished.fetch_add(1, Ordering::SeqCst);
        })
    }
}

/// Polls of a future that had already panicked, counted by a panic hook
/// (which keeps them out of the test output), and the hook itself.
static POISONED_REPOLLS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

fn count_poisoned_repolls() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info
                .payload_as_str()
                .is_some_and(|message| message.contains("resumed after panicking"))
            {
                POISONED_REPOLLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
            } else {
                previous(info);
            }
        }));
    });
}

// DRAIN-PANIC-01 (review t_bbffedfb): a retirement unit whose future panics
// must not strand the units queued behind it, nor be polled again. Before the
// repair, the dropped drain put the panicked callback back at the queue's
// head; every later drain resumed it ("`async fn` resumed after panicking"),
// panicked again and never reached the rest, so the store's retirement never
// settled again.
#[nativelink_test]
async fn a_panicking_retirement_unit_does_not_strand_the_queue() -> Result<(), Error> {
    use core::sync::atomic::Ordering;
    use nativelink_util::store_trait::StoreDriver;
    const N: usize = 40;
    let _serial = SERIAL.lock().await;
    count_poisoned_repolls();
    let repolls_before = POISONED_REPOLLS.load(Ordering::SeqCst);
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    let callback = Arc::new(PanicsOnce::default());
    store.clone().register_remove_callback(callback.clone())?;

    // An eviction batch: its first callback unit panics.
    write(&store, N).await?;
    eventually("the callback panicked", || {
        callback.panicked.load(Ordering::SeqCst)
    })
    .await;
    // Later evictions enqueue more work and start drains.
    for index in N + 1..N + 4 {
        write(&store, index).await?;
    }
    settle(&store).await;

    let settled = store.retirement_accounting();
    assert_eq!(
        (settled.queued_units, settled.queued_bytes),
        (0, 0),
        "{settled:?}"
    );
    assert_eq!(settled.panicked_units_total, 1, "{settled:?}");
    assert!(
        callback.finished.load(Ordering::SeqCst) > 0,
        "the units behind the panicked one ran"
    );
    for victim in &digests[..N / 2 - 1] {
        assert_eq!(content_files_of(&dirs, *victim).await, Vec::<String>::new());
    }
    assert_eq!(POISONED_REPOLLS.load(Ordering::SeqCst), repolls_before);
    assert!(logs_contain(
        "Filesystem store retirement unit panicked; the drain abandons it and continues"
    ));
    assert!(logs_contain("pending_unit=\"callback\""));
    Ok(())
}

// DRAIN-PANIC-01 with the optional cap: a write already asleep waiting for
// space held by a batch whose callback then panics still gets its space,
// without the panicked callback being polled again. Before the repair, the
// dropped drain's wake-up made the waiter restart a drain that resumed the
// panicked callback at the head, repeatedly, and the space never came.
#[nativelink_test]
async fn a_reservation_waiting_behind_a_panicking_unit_gets_its_space() -> Result<(), Error> {
    use core::sync::atomic::Ordering;
    use nativelink_util::store_trait::StoreDriver;
    const N: usize = 40;
    let _serial = SERIAL.lock().await;
    count_poisoned_repolls();
    let repolls_before = POISONED_REPOLLS.load(Ordering::SeqCst);
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    // Registered first, so it runs before the panicking callback. The
    // store's own executable-variant callback, ahead of both, takes no
    // admission for a digest that never had a variant, so it cannot hold
    // the batch.
    let blocker = Arc::new(BlocksFirstCall::default());
    store.clone().register_remove_callback(blocker.clone())?;
    let callback = Arc::new(PanicsOnce::default());
    store.clone().register_remove_callback(callback.clone())?;

    // The batch is held by its first callback, and its unrefs at the
    // retirement admission, until released.
    let admission = store.hold_retirement_admission_for_test().await;
    write(&store, N).await?;
    let held = store.retirement_accounting();
    assert!(held.draining && held.queued_bytes > SIZE as u64, "{held:?}");
    store.set_retirement_reservation_cap(Some(SIZE as u64));
    let waiter = nativelink_util::background_spawn!("waiting_write", {
        let store = store.clone();
        async move { write(&store, N + 1).await }
    });
    eventually("the write to wait", || {
        store.retirement_accounting().waiting_reservations == 1
    })
    .await;
    assert!(!callback.panicked.load(Ordering::SeqCst));

    // Released: the batch runs, its callback panics, and the rest must still
    // free the space the write waits for.
    blocker.release.add_permits(1);
    drop(admission);
    let written = timeout(WAIT, waiter)
        .await
        .expect("the waiting write never got its space")
        .expect("join")?;
    assert!(callback.panicked.load(Ordering::SeqCst));
    settle(&store).await;
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
    assert!(callback.finished.load(Ordering::SeqCst) > 0);
    for victim in &digests[..N / 2 - 1] {
        assert_eq!(content_files_of(&dirs, *victim).await, Vec::<String>::new());
    }
    assert_eq!(
        store.get_part_unchunked(written, 0, None).await?,
        blob(N + 1).1
    );
    assert_eq!(POISONED_REPOLLS.load(Ordering::SeqCst), repolls_before);
    Ok(())
}

// Per-waiter diagnostic port (nix-config 530c24d7): a retirement that waits
// for the admission past the warning threshold is reported once per holder,
// naming the waiter and the holder; that holder's release is reported too,
// with the same operation id.
#[nativelink_test]
async fn a_long_admission_wait_is_reported_once_per_holder_with_its_release() -> Result<(), Error> {
    use nativelink_util::store_trait::StoreKey;
    const N: usize = 40;
    const WAITING: &str =
        "Filesystem retirement admission is still waiting; holder is a point-in-time snapshot";
    const RELEASED: &str = "Reported filesystem retirement admission holder released";
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    store.set_retirement_wait_warning_for_test(
        Duration::from_millis(200),
        Duration::from_millis(20),
    );

    // The first removal's rename holds the admission until released.
    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let entered_tx = std::sync::Mutex::new(Some(entered_tx));
    let release_rx = std::sync::Mutex::new(Some(release_rx));
    store.set_after_retire_rename_hook_for_test(Some(Arc::new(move || {
        let entered = entered_tx.lock().unwrap().take();
        if let Some(tx) = entered {
            tx.send(()).unwrap();
            let release = release_rx.lock().unwrap().take();
            if let Some(rx) = release {
                rx.recv().unwrap();
            }
        }
    })));
    let remove = |digest: DigestInfo| {
        let map = store.get_evicting_map();
        tokio::spawn(async move { map.remove(&StoreKey::Digest(digest)).await })
    };
    let holder = remove(digests[N - 2]);
    tokio::task::spawn_blocking(move || entered_rx.recv_timeout(WAIT))
        .await
        .expect("join")
        .expect("the removal's rename reached its hook");
    // Two more removals wait for the admission it holds.
    let waiters = [remove(digests[N - 3]), remove(digests[N - 4])];

    timeout(WAIT, async {
        while !logs_contain(WAITING) {
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("no waiting warning while the admission was held");
    // Several more checks pass while the same holder still holds it.
    sleep(Duration::from_millis(300)).await;
    logs_assert(|lines: &[&str]| {
        let warnings: Vec<&&str> = lines.iter().filter(|line| line.contains(WAITING)).collect();
        if warnings.len() != 1 {
            return Err(format!(
                "{} waiting warnings for one holder",
                warnings.len()
            ));
        }
        let warning = warnings[0];
        if !(warning.contains(" WARN ")
            && warning.contains("operation_id=")
            && (warning.contains("kind=Rename") || warning.contains("kind=Variant"))
            && warning.contains("waited_ms=")
            && warning.contains("holder=(")
            && warning.contains("Rename"))
        {
            return Err(format!("incomplete waiting warning: {warning}"));
        }
        if lines.iter().any(|line| line.contains(RELEASED)) {
            return Err("a release reported while the holder still held".to_string());
        }
        Ok(())
    });
    assert!(
        !holder.is_finished(),
        "the holder still holds when reported"
    );

    release_tx.send(()).unwrap();
    assert!(
        timeout(WAIT, holder)
            .await
            .expect("holder finished")
            .expect("join")
    );
    for waiter in waiters {
        assert!(
            timeout(WAIT, waiter)
                .await
                .expect("waiter finished")
                .expect("join")
        );
    }
    timeout(WAIT, async {
        while !logs_contain(RELEASED) {
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the reported holder's release was not reported");
    // The release names the holder the waiting warning named.
    logs_assert(|lines: &[&str]| {
        let named = |line: &str, prefix: &str| {
            line.split(prefix)
                .nth(1)
                .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
                .map(str::to_string)
        };
        let warned = lines
            .iter()
            .find(|line| line.contains(WAITING))
            .and_then(|line| named(line, "holder=("));
        let released = lines
            .iter()
            .find(|line| line.contains(RELEASED))
            .and_then(|line| named(line, "holder_operation_id="));
        match (warned, released) {
            (Some(warned), Some(released)) if warned == released && !warned.is_empty() => Ok(()),
            other => Err(format!("warned and released holders differ: {other:?}")),
        }
    });
    store.set_after_retire_rename_hook_for_test(None);
    settle(&store).await;
    let settled = store.retirement_accounting();
    assert_eq!((settled.queued_units, settled.queued_bytes), (0, 0));
    Ok(())
}

/// Holds the first `count` admitted retirement operations of `store` (of any
/// kind), each until its release is sent; `entered` receives each one's index
/// as it is held.
fn hold_retirement_operations(
    store: &Arc<FilesystemStore>,
    count: usize,
) -> (
    std::sync::mpsc::Receiver<usize>,
    Vec<std::sync::mpsc::Sender<()>>,
) {
    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<usize>();
    let (releases, gates): (Vec<_>, Vec<_>) = (0..count)
        .map(|_| {
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            (tx, Some(rx))
        })
        .unzip();
    let gates = std::sync::Mutex::new(gates);
    let entered_tx = std::sync::Mutex::new(entered_tx);
    let next = core::sync::atomic::AtomicUsize::new(0);
    store.set_before_retirement_operation_hook_for_test(Some(Arc::new(move || {
        let index = next.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        let gate = gates.lock().unwrap().get_mut(index).and_then(Option::take);
        if let Some(gate) = gate {
            entered_tx.lock().unwrap().send(index).unwrap();
            gate.recv().unwrap();
        }
    })));
    (entered_rx, releases)
}

/// Pauses the store's waiting-retirement check once, right after it selects a
/// waiter, once armed: `selected` is signalled and the check waits for
/// `resume`.
fn pause_waiter_selection_once(
    store: &Arc<FilesystemStore>,
) -> (
    Arc<core::sync::atomic::AtomicBool>,
    std::sync::mpsc::Receiver<()>,
    std::sync::mpsc::Sender<()>,
) {
    let armed = Arc::new(core::sync::atomic::AtomicBool::new(false));
    let (selected_tx, selected_rx) = std::sync::mpsc::channel::<()>();
    let (resume_tx, resume_rx) = std::sync::mpsc::channel::<()>();
    let selected_tx = std::sync::Mutex::new(selected_tx);
    let resume_rx = std::sync::Mutex::new(resume_rx);
    let hook_armed = armed.clone();
    store.set_retirement_wait_selected_hook_for_test(Some(Arc::new(move || {
        if hook_armed.swap(false, core::sync::atomic::Ordering::SeqCst) {
            selected_tx.lock().unwrap().send(()).unwrap();
            resume_rx.lock().unwrap().recv().unwrap();
        }
    })));
    (armed, selected_rx, resume_tx)
}

const WAITING_WARNING: &str =
    "Filesystem retirement admission is still waiting; holder is a point-in-time snapshot";

/// (waiter operation id, holder operation id) of each waiting warning.
fn waiting_warnings(lines: &[&str]) -> Vec<(u64, u64)> {
    let number = |line: &str, prefix: &str| {
        line.split(prefix)
            .nth(1)
            .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|digits| digits.parse::<u64>().ok())
    };
    lines
        .iter()
        .filter(|line| line.contains(WAITING_WARNING))
        .map(|line| {
            (
                number(line, " operation_id=").expect("waiter id"),
                number(line, "holder=(").expect("holder id"),
            )
        })
        .collect()
}

/// Every waiting warning names a waiter admitted after its holder (the
/// admission is FIFO and ids are taken on arrival), so never the holder itself
/// or a waiter that was already gone; at most one per holder; and one names
/// the given holder. A macro: `logs_assert` exists only in a test's body.
macro_rules! assert_waiting_warnings_coherent {
    ($holder:expr) => {{
        let holder: u64 = $holder;
        logs_assert(|lines: &[&str]| {
            let warnings = waiting_warnings(lines);
            if let Some(bad) = warnings.iter().find(|(waiter, holder)| waiter <= holder) {
                return Err(format!(
                    "a waiting warning names a stale or self waiter: {bad:?} in {warnings:?}"
                ));
            }
            let mut holders: Vec<u64> = warnings.iter().map(|(_, holder)| *holder).collect();
            holders.sort_unstable();
            if holders.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(format!("two waiting warnings for one holder: {warnings:?}"));
            }
            if holders.contains(&holder) {
                Ok(())
            } else {
                Err(format!(
                    "no waiting warning for holder {holder}: {warnings:?}"
                ))
            }
        });
    }};
}

/// The holder id of the `count`th waiting warning, once there is one. A macro:
/// `logs_assert` exists only in a test's body.
macro_rules! nth_waiting_holder {
    ($count:expr) => {{
        let count: usize = $count;
        timeout(WAIT, async {
            loop {
                let found = core::cell::Cell::new(None);
                logs_assert(|lines: &[&str]| {
                    let warnings = waiting_warnings(lines);
                    if warnings.len() >= count {
                        found.set(Some(warnings[count - 1].1));
                    }
                    Ok(())
                });
                if let Some(holder) = found.get() {
                    break holder;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("waiting warning")
    }};
}

// DIAG-RACE-01 (review t_e5c79107): the selected waiter is admitted, and the
// holder changes, in the gap after selection. No warning may name the
// admitted waiter as waiting on itself, nor use up the new holder's one
// warning: the real waiter behind the new holder is still reported.
#[nativelink_test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waiter_admitted_after_selection_is_not_reported_waiting_on_itself() -> Result<(), Error>
{
    use nativelink_util::store_trait::StoreKey;
    const N: usize = 40;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    store.set_retirement_wait_warning_for_test(
        Duration::from_millis(200),
        Duration::from_millis(20),
    );
    // Each removal unlinks its digest's executable variant and then renames
    // its file: two admitted operations. A digest without a variant would
    // take only the rename.
    for digest in &digests[N - 3..N - 1] {
        store.get_executable_hardlink_source(digest).await?;
    }
    let (entered, releases) = hold_retirement_operations(&store, 2);
    let remove = |digest: DigestInfo| {
        let map = store.get_evicting_map();
        tokio::spawn(async move { map.remove(&StoreKey::Digest(digest)).await })
    };
    let first = remove(digests[N - 2]);
    let entered = tokio::task::spawn_blocking(move || {
        assert_eq!(entered.recv_timeout(WAIT), Ok(0), "the first holder");
        entered
    })
    .await
    .expect("join");
    let second = remove(digests[N - 3]);
    let first_holder = nth_waiting_holder!(1);

    let (armed, selected, resume) = pause_waiter_selection_once(&store);
    armed.store(true, core::sync::atomic::Ordering::SeqCst);
    tokio::task::spawn_blocking(move || selected.recv_timeout(WAIT))
        .await
        .expect("join")
        .expect("a waiter was selected");
    // In the gap: the first holder releases, so the selected (longest, FIFO
    // first) waiter is admitted and becomes the next holder.
    releases[0].send(()).unwrap();
    sleep(Duration::from_millis(300)).await;
    resume.send(()).unwrap();
    let entered = tokio::task::spawn_blocking(move || entered.recv_timeout(WAIT))
        .await
        .expect("join");
    assert_eq!(entered, Ok(1), "the second holder");

    let second_holder = nth_waiting_holder!(2);
    assert_ne!(first_holder, second_holder);
    assert_waiting_warnings_coherent!(second_holder);

    releases[1].send(()).unwrap();
    for task in [first, second] {
        assert!(
            timeout(WAIT, task)
                .await
                .expect("removal finished")
                .expect("join")
        );
    }
    store.set_before_retirement_operation_hook_for_test(None);
    store.set_retirement_wait_selected_hook_for_test(None);
    settle(&store).await;
    assert_waiting_warnings_coherent!(second_holder);
    Ok(())
}

// DIAG-RACE-01 (review t_e5c79107): the selected waiter is cancelled, and the
// holder changes, in the gap after selection. The gone waiter may not be
// reported, nor use up the new holder's one warning: the real waiter behind
// the new holder is still reported.
#[nativelink_test(flavor = "multi_thread", worker_threads = 4)]
async fn a_waiter_cancelled_after_selection_is_not_reported() -> Result<(), Error> {
    use nativelink_util::store_trait::StoreKey;
    const N: usize = 40;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    store.set_retirement_wait_warning_for_test(
        Duration::from_millis(200),
        Duration::from_millis(20),
    );
    // Each removal unlinks its digest's executable variant and then renames
    // its file: two admitted operations. A digest without a variant would
    // take only the rename.
    for digest in &digests[N - 4..N - 1] {
        store.get_executable_hardlink_source(digest).await?;
    }
    let (entered, releases) = hold_retirement_operations(&store, 2);
    let remove = |digest: DigestInfo| {
        let map = store.get_evicting_map();
        tokio::spawn(async move { map.remove(&StoreKey::Digest(digest)).await })
    };
    // The first removal's first operation (its variant) holds; the second
    // removal's first operation waits, the only waiter.
    let first = remove(digests[N - 2]);
    let entered = tokio::task::spawn_blocking(move || {
        assert_eq!(entered.recv_timeout(WAIT), Ok(0), "the first holder");
        entered
    })
    .await
    .expect("join");
    let second = remove(digests[N - 3]);
    let first_holder = nth_waiting_holder!(1);

    let (armed, selected, resume) = pause_waiter_selection_once(&store);
    armed.store(true, core::sync::atomic::Ordering::SeqCst);
    tokio::task::spawn_blocking(move || selected.recv_timeout(WAIT))
        .await
        .expect("join")
        .expect("a waiter was selected");
    // In the gap: the selected waiter's removal is cancelled while it waits,
    // then the first holder releases, and the first removal's next operation
    // (its rename) arrives and can take the admission: a new holder.
    second.abort();
    sleep(Duration::from_millis(100)).await;
    releases[0].send(()).unwrap();
    sleep(Duration::from_millis(300)).await;
    resume.send(()).unwrap();
    let entered = tokio::task::spawn_blocking(move || entered.recv_timeout(WAIT))
        .await
        .expect("join");
    assert_eq!(entered, Ok(1), "the second holder");
    // A real waiter behind the second holder.
    let third = remove(digests[N - 4]);

    let second_holder = nth_waiting_holder!(2);
    assert_ne!(first_holder, second_holder);
    assert_waiting_warnings_coherent!(second_holder);

    releases[1].send(()).unwrap();
    let _cancelled = second.await;
    for task in [first, third] {
        assert!(
            timeout(WAIT, task)
                .await
                .expect("removal finished")
                .expect("join")
        );
    }
    store.set_before_retirement_operation_hook_for_test(None);
    store.set_retirement_wait_selected_hook_for_test(None);
    settle(&store).await;
    assert_waiting_warnings_coherent!(second_holder);
    Ok(())
}

// Watcher runtime loss (review t_6a4d226c): the store's watch task runs on the
// runtime that started it and ends with it. A store opened on a runtime that
// then shuts down, and used on another, must still report a long admission
// wait: the waiting retirement restarts the watch task on its own runtime,
// and the restart is logged. Before, the watch task was started only when the
// store was opened, and nothing was reported.
#[nativelink_test]
async fn a_long_admission_wait_is_reported_after_the_watch_runtime_shut_down() -> Result<(), Error>
{
    use nativelink_util::store_trait::StoreKey;
    const N: usize = 40;
    const RESTARTED: &str = "Filesystem retirement wait watch task had ended while the store stayed in use; restarted on the current runtime";
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    // Opened and filled on a runtime of its own, which then shuts down with
    // the store's watch task.
    let (content, temp) = (dirs.content.clone(), dirs.temp.clone());
    #[expect(
        clippy::disallowed_methods,
        reason = "the store must be opened on a runtime that then shuts down"
    )]
    let opener = std::thread::spawn(move || {
        let runtime = own_runtime();
        let opened = run_on(&runtime, async move {
            let dirs = Dirs { content, temp };
            let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
            let digests = fill(&store, N - 1).await?;
            Ok::<_, Error>((store, digests))
        });
        drop(runtime);
        opened
    });
    let (store, digests) = tokio::task::spawn_blocking(move || opener.join())
        .await
        .expect("join")
        .expect("the opening thread panicked")?;
    store.set_retirement_wait_warning_for_test(
        Duration::from_millis(200),
        Duration::from_millis(20),
    );

    // As in the per-waiter test: the first removal's rename holds the
    // admission until released, and two more removals wait for it.
    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let entered_tx = std::sync::Mutex::new(Some(entered_tx));
    let release_rx = std::sync::Mutex::new(Some(release_rx));
    store.set_after_retire_rename_hook_for_test(Some(Arc::new(move || {
        let entered = entered_tx.lock().unwrap().take();
        if let Some(tx) = entered {
            tx.send(()).unwrap();
            let release = release_rx.lock().unwrap().take();
            if let Some(rx) = release {
                rx.recv().unwrap();
            }
        }
    })));
    let remove = |digest: DigestInfo| {
        let map = store.get_evicting_map();
        nativelink_util::background_spawn!("removal", async move {
            map.remove(&StoreKey::Digest(digest)).await
        })
    };
    let holder = remove(digests[N - 2]);
    tokio::task::spawn_blocking(move || entered_rx.recv_timeout(WAIT))
        .await
        .expect("join")
        .expect("the removal's rename reached its hook");
    let waiters = [remove(digests[N - 3]), remove(digests[N - 4])];

    timeout(WAIT, async {
        while !logs_contain(WAITING_WARNING) {
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("no waiting warning after the watch runtime shut down");
    assert!(logs_contain(RESTARTED));
    logs_assert(|lines: &[&str]| {
        let restarts = lines.iter().filter(|line| line.contains(RESTARTED)).count();
        if restarts == 1 {
            Ok(())
        } else {
            Err(format!("{restarts} watch restarts, expected one"))
        }
    });
    assert!(
        !holder.is_finished(),
        "the holder still holds when reported"
    );

    release_tx.send(()).unwrap();
    assert!(
        timeout(WAIT, holder)
            .await
            .expect("holder finished")
            .expect("join")
    );
    for waiter in waiters {
        assert!(
            timeout(WAIT, waiter)
                .await
                .expect("waiter finished")
                .expect("join")
        );
    }
    store.set_after_retire_rename_hook_for_test(None);
    settle(&store).await;
    let settled = store.retirement_accounting();
    assert_eq!((settled.queued_units, settled.queued_bytes), (0, 0));
    Ok(())
}

// WATCH-R3-01 (review t_d0357215): retirements already waiting on a surviving
// runtime when the watch task's runtime shuts down must be reported again,
// with no further retirement started. Here the store is opened on runtime A,
// whose watch task is the store's; a holder and a waiter then wait on the
// test's runtime; only then does A shut down. The waiter, woken by the watch
// task's end, starts another on its own runtime. Before the repair, it had
// checked the watch task only once, when it started waiting, and nothing was
// reported until another retirement started.
#[nativelink_test]
async fn waiting_retirements_are_reported_after_the_watch_runtime_shuts_down_under_them()
-> Result<(), Error> {
    use nativelink_util::store_trait::StoreKey;
    const N: usize = 40;
    const RESTARTED: &str = "Filesystem retirement wait watch task had ended while the store stayed in use; restarted on the current runtime";
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    // Runtime A opens and fills the store, then keeps running (and with it
    // the store's watch task) until told to shut down.
    let (content, temp) = (dirs.content.clone(), dirs.temp.clone());
    let (opened_tx, opened_rx) = std::sync::mpsc::channel();
    let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
    #[expect(
        clippy::disallowed_methods,
        reason = "the store must be opened on a runtime that shuts down later"
    )]
    let runtime_a = std::thread::spawn(move || {
        let runtime = own_runtime();
        let opened = run_on(&runtime, async move {
            let dirs = Dirs { content, temp };
            let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
            let digests = fill(&store, N - 1).await?;
            Ok::<_, Error>((store, digests))
        });
        opened_tx.send(opened).unwrap();
        shutdown_rx.recv().unwrap();
        drop(runtime);
    });
    let (store, digests) = tokio::task::spawn_blocking(move || opened_rx.recv())
        .await
        .expect("join")
        .expect("runtime A opened the store")?;
    // No wait is reported while A's watch task runs: that would use up the
    // holder's one warning before the test's.
    store
        .set_retirement_wait_warning_for_test(Duration::from_secs(3600), Duration::from_millis(20));

    let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let entered_tx = std::sync::Mutex::new(Some(entered_tx));
    let release_rx = std::sync::Mutex::new(Some(release_rx));
    store.set_after_retire_rename_hook_for_test(Some(Arc::new(move || {
        let entered = entered_tx.lock().unwrap().take();
        if let Some(tx) = entered {
            tx.send(()).unwrap();
            let release = release_rx.lock().unwrap().take();
            if let Some(rx) = release {
                rx.recv().unwrap();
            }
        }
    })));
    let remove = |digest: DigestInfo| {
        let map = store.get_evicting_map();
        nativelink_util::background_spawn!("removal", async move {
            map.remove(&StoreKey::Digest(digest)).await
        })
    };
    let holder = remove(digests[N - 2]);
    tokio::task::spawn_blocking(move || entered_rx.recv_timeout(WAIT))
        .await
        .expect("join")
        .expect("the removal's rename reached its hook");
    let waiter = remove(digests[N - 3]);
    eventually("the waiter to wait", || {
        store.waiting_retirements_for_test() == 1
    })
    .await;

    // A shuts down under the waiting retirement; nothing else starts one.
    shutdown_tx.send(()).unwrap();
    tokio::task::spawn_blocking(move || runtime_a.join())
        .await
        .expect("join")
        .expect("runtime A's thread panicked");
    store.set_retirement_wait_warning_for_test(
        Duration::from_millis(200),
        Duration::from_millis(20),
    );
    timeout(WAIT, async {
        while !logs_contain(WAITING_WARNING) {
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("no waiting warning after the watch runtime shut down under the waiter");
    logs_assert(|lines: &[&str]| {
        let restarts = lines.iter().filter(|line| line.contains(RESTARTED)).count();
        if restarts == 1 {
            Ok(())
        } else {
            Err(format!("{restarts} watch restarts, expected one"))
        }
    });
    assert_eq!(store.waiting_retirements_for_test(), 1);
    assert!(
        !holder.is_finished(),
        "the holder still holds when reported"
    );
    assert!(
        !waiter.is_finished(),
        "the waiter still waits when reported"
    );
    // Several more checks pass: still one warning, for the one holder.
    sleep(Duration::from_millis(300)).await;
    let reported_holder = nth_waiting_holder!(1);
    assert_waiting_warnings_coherent!(reported_holder);
    logs_assert(|lines: &[&str]| {
        let warnings = waiting_warnings(lines).len();
        if warnings == 1 {
            Ok(())
        } else {
            Err(format!("{warnings} waiting warnings for one holder"))
        }
    });

    release_tx.send(()).unwrap();
    for removal in [holder, waiter] {
        assert!(
            timeout(WAIT, removal)
                .await
                .expect("removal finished")
                .expect("join")
        );
    }
    assert_eq!(store.waiting_retirements_for_test(), 0);
    store.set_after_retire_rename_hook_for_test(None);
    settle(&store).await;
    let settled = store.retirement_accounting();
    assert_eq!((settled.queued_units, settled.queued_bytes), (0, 0));
    Ok(())
}

/// The names of the content directory's files with their `-{generation}`
/// suffix removed, that is the digests the store holds files for.
async fn content_digests(dirs: &Dirs) -> std::collections::HashSet<String> {
    let mut digests = std::collections::HashSet::new();
    let mut entries = tokio::fs::read_dir(format!("{}/{DIGEST_FOLDER_V2}", dirs.content))
        .await
        .expect("content directory");
    while let Some(entry) = entries.next_entry().await.expect("read content directory") {
        let name = entry.file_name().into_string().expect("utf-8 file name");
        let (digest, _generation) = name.rsplit_once('-').expect("a generation suffix");
        digests.insert(digest.to_string());
    }
    digests
}

/// Checks that the first `N / 2 - 1` of `digests` (the oldest, all evicted)
/// have no content file left and that the newest is still readable.
async fn assert_victims_retired(
    store: &Arc<FilesystemStore>,
    dirs: &Dirs,
    digests: &[DigestInfo],
    n: usize,
) -> Result<(), Error> {
    let held = content_digests(dirs).await;
    for victim in &digests[..n / 2 - 1] {
        assert!(
            !held.contains(&victim.to_string()),
            "evicted {victim} still has its content file"
        );
    }
    let newest = n - 2;
    assert_eq!(
        store.get_part_unchunked(digests[newest], 0, None).await?,
        blob(newest).1
    );
    Ok(())
}

// RETIRE-BATCH-01: an eviction batch is retired in a few admitted retirement
// operations (blocking-pool tasks), not one or two per evicted entry. Before
// the repair each entry's rename was its own operation and so was the
// unlink of its executable variant, although it never had one: this batch
// took 2,050 blocking-pool round trips. On the workstation's worker
// that serial drain retired about 1,800 files a second, slower than they were
// inserted, and a backlog of 136 GiB of evicted files built up on its tmpfs.
#[nativelink_test]
async fn an_eviction_batch_is_renamed_in_few_retirement_operations() -> Result<(), Error> {
    use core::sync::atomic::{AtomicUsize, Ordering};
    const N: usize = 2048;
    let _serial = SERIAL.lock().await;
    let dirs = dirs();
    let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
    let digests = fill(&store, N - 1).await?;
    let operations = Arc::new(AtomicUsize::new(0));
    store.set_before_retirement_operation_hook_for_test(Some(Arc::new({
        let operations = operations.clone();
        move || {
            operations.fetch_add(1, Ordering::SeqCst);
        }
    })));

    // This insertion crosses max_bytes and evicts about half of the store.
    write(&store, N).await?;
    settle(&store).await;
    store.set_before_retirement_operation_hook_for_test(None);

    let settled = store.retirement_accounting();
    assert_eq!(
        (
            settled.queued_units,
            settled.queued_bytes,
            settled.detached_total
        ),
        (0, 0, 0),
        "{settled:?}"
    );
    assert_victims_retired(&store, &dirs, &digests, N).await?;
    let operations = operations.load(Ordering::SeqCst);
    let evicted = N / 2 - 1;
    assert!(
        operations > 0 && operations * 64 <= evicted,
        "{evicted} evictions took {operations} retirement operations"
    );
    Ok(())
}

/// A runtime whose blocking pool is small, for a test that keeps it busy.
#[expect(
    clippy::disallowed_methods,
    reason = "the test needs a runtime with a small blocking pool of its own"
)]
fn small_blocking_pool_runtime(max_blocking_threads: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(max_blocking_threads)
        .enable_all()
        .build()
        .expect("runtime")
}

// RETIRE-BATCH-02: with the blocking pool busy, as on a worker running 256
// actions, an eviction batch still drains quickly. Each submission to the
// pool waits behind the other tasks queued there (here, about 3 ms); before
// the repair the drain paid that wait once per evicted file, so 1,023
// evictions took about 4 s, and now it pays it once per batch.
#[test]
fn an_eviction_batch_drains_quickly_while_the_blocking_pool_is_busy() -> Result<(), Error> {
    use core::sync::atomic::{AtomicBool, Ordering};
    const N: usize = 2048;
    const BLOCKING_THREADS: usize = 4;
    const LOAD_TASKS: usize = 12;
    const LOAD_SLEEP: Duration = Duration::from_millis(2);
    // One round trip per evicted file takes over 2 s here; batched, well
    // under 0.1 s.
    const BOUND: Duration = Duration::from_millis(1500);
    let _serial = SERIAL.blocking_lock();
    let dirs = dirs();
    let runtime = small_blocking_pool_runtime(BLOCKING_THREADS);
    run_on(&runtime, async {
        let store = open_store(&dirs, N * SIZE, N * SIZE / 2).await?;
        let digests = fill(&store, N - 1).await?;

        let stop = Arc::new(AtomicBool::new(false));
        let load: Vec<_> = (0..LOAD_TASKS)
            .map(|_| {
                let stop = stop.clone();
                nativelink_util::background_spawn!("blocking_pool_load", async move {
                    while !stop.load(Ordering::Relaxed) {
                        nativelink_util::spawn_blocking!("blocking_pool_load_sleep", || {
                            std::thread::sleep(LOAD_SLEEP);
                        })
                        .await
                        .expect("load task");
                    }
                })
            })
            .collect();
        // Let the load fill the pool's queue.
        sleep(Duration::from_millis(50)).await;

        write(&store, N).await?;
        let started = std::time::Instant::now();
        timeout(
            Duration::from_secs(120),
            store.retirement_settled_for_test(),
        )
        .await
        .expect("retirement did not settle");
        let drained = started.elapsed();

        stop.store(true, Ordering::Relaxed);
        for task in load {
            task.await.expect("load task");
        }
        assert_victims_retired(&store, &dirs, &digests, N).await?;
        let evicted = N / 2 - 1;
        eprintln!(
            "{evicted} evictions drained in {} ms under a busy blocking pool",
            drained.as_millis()
        );
        assert!(
            drained < BOUND,
            "{evicted} evictions took {} ms to drain under a busy blocking pool (bound {} ms)",
            drained.as_millis(),
            BOUND.as_millis()
        );
        Ok(())
    })
}
