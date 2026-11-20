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
#[nativelink_test]
async fn a_long_drain_reports_its_backlog() -> Result<(), Error> {
    const N: usize = 40;
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

    // A batch whose drain runs past a zero interval reports while it runs.
    store.set_retirement_report_interval_for_test(Duration::ZERO);
    let admission = store.hold_retirement_admission_for_test().await;
    for index in N + 1..2 * N {
        write(&store, index).await?;
    }
    let held = store.retirement_accounting();
    assert!(held.queued_units > 0, "a batch is queued: {held:?}");
    drop(admission);
    settle(&store).await;
    let store_field = format!("store={}", dirs.content);
    logs_assert(|lines: &[&str]| {
        let backlog = lines.iter().any(|line| {
            line.contains(" INFO ")
                && line.contains("Filesystem store retirement backlog")
                && line.contains("queued_units=")
                && line.contains(&store_field)
        });
        let caught_up = lines.iter().any(|line| {
            line.contains(" INFO ") && line.contains("Filesystem store retirement drain caught up")
        });
        if backlog && caught_up {
            Ok(())
        } else {
            Err(format!(
                "backlog report {backlog}, caught-up report {caught_up}"
            ))
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
    assert_eq!(
        after.queued_units,
        running.queued_units + 1,
        "the running unit was put back"
    );
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
