//! Asynchronous byte-range I/O.
//!
//! [`AsyncRangeReader`] is the core I/O abstraction: remote sources (S3, HTTP) implement it
//! natively with non-blocking requests, so waiting on the network never occupies a thread.
//! The synchronous [`RangeReader`] trait stays for plain-thread users and custom readers, and
//! the adapters here convert in both directions:
//!
//! * [`SyncToAsync`] runs a blocking [`RangeReader`] on tokio's blocking pool (in-memory
//!   readers are served inline).
//! * [`AsyncToSync`] exposes an async reader as a [`RangeReader`]. The future is spawned onto a
//!   private, crate-owned runtime and the caller blocks on its completion, so it is safe from
//!   plain threads, `spawn_blocking` threads and (wastefully, but without panic or deadlock)
//!   from inside another runtime's worker thread.
//!
//! [`fetch_ranges`] fetches many ranges concurrently, merging nearby ones into one request.

use std::ops::Range;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::range_reader::RangeReader;
use crate::tiff_utils::AnyResult;

/// Tuning knobs for remote I/O.
///
/// The defaults suit a tile server reading public COGs over S3/HTTPS. Pass a customised value
/// to [`CogReader::open_async_with_options`](crate::CogReader::open_async_with_options).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IoOptions {
    /// Maximum simultaneous HTTP requests issued for one fetch (one output tile).
    pub max_concurrent_requests: usize,
    /// Maximum simultaneous HTTP requests per store (bucket or host), across all fetches.
    pub max_in_flight_per_store: usize,
    /// Ranges separated by at most this many bytes are merged into one request.
    pub coalesce_gap: u64,
    /// A merged request is never grown beyond this many bytes (a single larger range is
    /// still fetched whole).
    pub max_merged_len: u64,
    /// Retries after a failed request (`object_store` retries 5xx, timeouts and connection
    /// errors with exponential backoff).
    pub max_retries: usize,
    /// No retry is started once this much time has passed since the request began.
    pub retry_timeout: Duration,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
    /// Per-request timeout.
    pub request_timeout: Duration,
}

impl Default for IoOptions {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 16,
            max_in_flight_per_store: 128,
            coalesce_gap: 32 * 1024,
            max_merged_len: 4 * 1024 * 1024,
            max_retries: 3,
            retry_timeout: Duration::from_secs(15),
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(20),
        }
    }
}

static DEFAULT_IO_OPTIONS: LazyLock<IoOptions> = LazyLock::new(IoOptions::default);

/// Asynchronous random-access reader over one object (local file, memory, S3, HTTP, ...).
///
/// Implementations must be cheap to share (`Arc<dyn AsyncRangeReader>`) and safe to call
/// concurrently. The futures are boxed rather than `async fn` so the trait stays
/// object-safe.
pub trait AsyncRangeReader: Send + Sync {
    /// Read exactly `len` bytes starting at `offset`.
    ///
    /// # Errors
    /// Returns an error on I/O or network failure, or if the range lies outside the object.
    fn read_range(&self, offset: u64, len: usize) -> BoxFuture<'_, AnyResult<Bytes>>;

    /// Read several ranges, fetching them concurrently and merging nearby ones.
    ///
    /// Results are returned in the order of `ranges`. The default implementation uses
    /// [`fetch_ranges`] with [`io_options`](Self::io_options).
    ///
    /// # Errors
    /// Returns the first read error; remaining requests are cancelled.
    fn read_ranges<'a>(&'a self, ranges: &'a [Range<u64>]) -> BoxFuture<'a, AnyResult<Vec<Bytes>>> {
        Box::pin(fetch_ranges(self, ranges, self.io_options()))
    }

    /// Total size of the object in bytes.
    fn size(&self) -> u64;

    /// Human-readable identifier (URL or path).
    fn identifier(&self) -> &str;

    /// Token that changes when the object's content changes: the ETag (or size and modification
    /// time) of a remote object, size and modification time of a local file; `None` when the
    /// source has no notion of version (in-memory data, mocks).
    ///
    /// Together with [`identifier`](Self::identifier) it keys decoded tiles, so a replaced
    /// object can never be served from tiles decoded from its predecessor.
    fn version(&self) -> Option<&str> {
        None
    }

    /// True for fast random access (local files, memory); false for network sources.
    fn is_local(&self) -> bool {
        let id = self.identifier();
        !id.starts_with("http://") && !id.starts_with("https://") && !id.starts_with("s3://")
    }

    /// Options governing [`read_ranges`](Self::read_ranges).
    fn io_options(&self) -> &IoOptions {
        &DEFAULT_IO_OPTIONS
    }

    /// Open the same object again, fresh from the source (new size, version and prefix).
    ///
    /// Used to recover after a read reported [`SourceChanged`]. The default implementation fails:
    /// sources that cannot be replaced (memory, mocks) have nothing to reopen.
    fn reopen(&self) -> BoxFuture<'_, AnyResult<Arc<dyn AsyncRangeReader>>> {
        let identifier = self.identifier().to_string();
        Box::pin(async move { Err(format!("{identifier} cannot be reopened").into()) })
    }
}

/// Merge `ranges` into a sorted list of disjoint requests.
///
/// Ranges closer than `gap` bytes are merged, unless the merged request would exceed `max_len`.
/// The result is sorted by start and covers every input range.
pub(crate) fn merge_ranges(ranges: &[Range<u64>], gap: u64, max_len: u64) -> Vec<Range<u64>> {
    let mut sorted: Vec<Range<u64>> = ranges.iter().filter(|r| r.start < r.end).cloned().collect();
    sorted.sort_unstable_by_key(|r| (r.start, r.end));
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(sorted.len());
    for r in sorted {
        if let Some(last) = merged.last_mut()
            && r.start <= last.end.saturating_add(gap)
            && r.end.max(last.end) - last.start <= max_len
        {
            last.end = last.end.max(r.end);
        } else {
            merged.push(r);
        }
    }
    merged
}

/// The object was replaced or modified after it was opened: a conditional read (`If-Match` /
/// `If-Unmodified-Since`) failed with `412 Precondition Failed`.
///
/// The reader that returned it still describes the old version; open the source again (the
/// extraction and point-query entry points do so once, automatically).
/// [`is_source_changed`] recognises it inside the errors this crate returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChanged {
    /// Identifier of the source (URL or path).
    pub identifier: String,
}

impl std::fmt::Display for SourceChanged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} was replaced or modified after it was opened", self.identifier)
    }
}

impl std::error::Error for SourceChanged {}

/// Whether `err` is, or wraps, a [`SourceChanged`] error.
#[must_use]
pub fn is_source_changed(err: &(dyn std::error::Error + 'static)) -> bool {
    err.is::<SourceChanged>() || err.downcast_ref::<RangeFetchError>().is_some_and(|e| e.source_changed)
}

/// A failed (possibly merged) range request, naming the byte range it covered. Returned (boxed)
/// by [`fetch_ranges`] so callers can tell which of their ranges was affected.
#[derive(Debug)]
pub struct RangeFetchError {
    /// The range that was requested (after merging).
    pub range: Range<u64>,
    message: String,
    /// The request failed because the object changed ([`SourceChanged`]).
    source_changed: bool,
}

impl std::fmt::Display for RangeFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RangeFetchError {}

/// Fetch `ranges` from `reader`: merge nearby ranges (see [`IoOptions::coalesce_gap`]), issue
/// up to [`IoOptions::max_concurrent_requests`] requests at a time, and slice the responses
/// back into the requested ranges (zero-copy). The first error cancels the rest.
///
/// # Errors
/// Returns the first request error, or an error if a response is shorter than requested.
pub async fn fetch_ranges<R: AsyncRangeReader + ?Sized>(
    reader: &R,
    ranges: &[Range<u64>],
    opts: &IoOptions,
) -> AnyResult<Vec<Bytes>> {
    let merged = merge_ranges(ranges, opts.coalesce_gap, opts.max_merged_len);
    let mut fetched: Vec<Option<Bytes>> = vec![None; merged.len()];
    let jobs: Vec<(usize, u64, u64)> = merged.iter().enumerate().map(|(i, m)| (i, m.start, m.end - m.start)).collect();
    let mut in_flight = stream::iter(jobs)
        .map(|(i, start, len)| async move {
            let range = start..start + len;
            let fail = |message: String| RangeFetchError { range: range.clone(), message, source_changed: false };
            let len = usize::try_from(len).map_err(|e| fail(format!("range too large: {e}")))?;
            let bytes = reader.read_range(start, len).await.map_err(|e| RangeFetchError {
                range: range.clone(),
                source_changed: is_source_changed(&*e),
                message: e.to_string(),
            })?;
            if bytes.len() != len {
                return Err(fail(format!(
                    "short read from {}: requested {len} bytes at offset {start}, got {}",
                    reader.identifier(),
                    bytes.len()
                )));
            }
            Ok::<_, RangeFetchError>((i, bytes))
        })
        .buffer_unordered(opts.max_concurrent_requests.max(1));
    while let Some((i, bytes)) = in_flight.try_next().await? {
        fetched[i] = Some(bytes);
    }
    drop(in_flight);

    ranges
        .iter()
        .map(|r| {
            if r.start >= r.end {
                return Ok(Bytes::new());
            }
            let idx = merged.partition_point(|m| m.start <= r.start) - 1;
            let base = merged[idx].start;
            let bytes = fetched[idx].as_ref().expect("every merged range was fetched");
            // Both bounds are within the merged range, whose length fits in usize.
            Ok(bytes.slice((r.start - base) as usize..(r.end - base) as usize))
        })
        .collect()
}

/// Adapter running a synchronous [`RangeReader`] as an [`AsyncRangeReader`].
///
/// Reads run on tokio's blocking pool, except for readers that report
/// [`RangeReader::reads_inline`] (in-memory data), which are served directly. Must be polled
/// inside a tokio runtime unless the reader is inline.
pub struct SyncToAsync {
    inner: Arc<dyn RangeReader>,
    inline: bool,
}

impl SyncToAsync {
    /// Wrap `inner`; reads use `spawn_blocking` unless `inner.reads_inline()`.
    #[must_use]
    pub fn new(inner: Arc<dyn RangeReader>) -> Self {
        let inline = inner.reads_inline();
        Self { inner, inline }
    }

    /// Wrap `inner` so every read runs directly in the polling thread. Used internally for
    /// the synchronous entry points, where the caller already is on a thread allowed to block.
    pub(crate) fn inline(inner: Arc<dyn RangeReader>) -> Self {
        Self { inner, inline: true }
    }
}

impl AsyncRangeReader for SyncToAsync {
    fn read_range(&self, offset: u64, len: usize) -> BoxFuture<'_, AnyResult<Bytes>> {
        if self.inline {
            let result = self.inner.read_range(offset, len).map(Bytes::from);
            return Box::pin(std::future::ready(result));
        }
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            tokio::task::spawn_blocking(move || inner.read_range(offset, len))
                .await
                .map_err(|e| format!("Task join error: {e}"))?
                .map(Bytes::from)
        })
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn identifier(&self) -> &str {
        self.inner.identifier()
    }

    fn version(&self) -> Option<&str> {
        self.inner.version()
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }
}

/// The crate's private I/O runtime.
///
/// All network requests run here (via [`spawn_io`]) and so do the futures of synchronous callers
/// (via [`block_on_io`]). HTTP connections are driven by tasks of the runtime that opened them, so
/// keeping every request on one runtime that lives for the whole process lets one HTTP client
/// per bucket or host be shared by callers on any number of runtimes, including short-lived ones.
static IO_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    // Network work is light (TLS, framing); a few threads saturate any link.
    let workers = std::thread::available_parallelism().map_or(2, |n| n.get().clamp(2, 8));
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .thread_name("cogrs-io")
        .enable_all()
        .build()
        .expect("failed to start the cogrs I/O runtime")
});

/// Aborts the task when dropped, so cancelling the caller cancels the request.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl<T> Future for AbortOnDrop<T> {
    type Output = Result<T, tokio::task::JoinError>;

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        std::pin::Pin::new(&mut self.0).poll(cx)
    }
}

/// Run `fut` on the private I/O runtime and await its result from any runtime.
///
/// Dropping the returned future aborts the spawned task.
pub(crate) async fn spawn_io<T: Send + 'static>(fut: impl Future<Output = T> + Send + 'static) -> AnyResult<T> {
    AbortOnDrop(IO_RUNTIME.spawn(fut)).await.map_err(|e| format!("I/O task failed: {e}").into())
}

/// Run `fut` to completion on the private I/O runtime, blocking the calling thread.
///
/// The future is *spawned* on the runtime (it never runs on the caller's thread), and the
/// caller waits on the join handle with a plain executor that needs no tokio context. That is
/// safe from any thread; on an async worker it blocks that worker but cannot deadlock.
pub(crate) fn block_on_io<T: Send + 'static>(fut: impl Future<Output = T> + Send + 'static) -> AnyResult<T> {
    let handle = IO_RUNTIME.spawn(fut);
    futures::executor::block_on(handle).map_err(|e| format!("I/O task failed: {e}").into())
}

/// Adapter exposing an [`AsyncRangeReader`] through the synchronous [`RangeReader`] trait.
///
/// Every call blocks the calling thread (see the module docs for where that is safe). Used by
/// the sync entry points such as [`CogReader::open`](crate::CogReader::open) for remote
/// sources.
pub struct AsyncToSync {
    inner: Arc<dyn AsyncRangeReader>,
}

impl AsyncToSync {
    /// Wrap `inner`.
    #[must_use]
    pub fn new(inner: Arc<dyn AsyncRangeReader>) -> Self {
        Self { inner }
    }

    /// The wrapped async reader.
    #[must_use]
    pub fn inner(&self) -> &Arc<dyn AsyncRangeReader> {
        &self.inner
    }
}

impl RangeReader for AsyncToSync {
    fn read_range(&self, offset: u64, length: usize) -> AnyResult<Vec<u8>> {
        let inner = Arc::clone(&self.inner);
        let bytes = block_on_io(async move { inner.read_range(offset, length).await })??;
        Ok(bytes.into())
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn identifier(&self) -> &str {
        self.inner.identifier()
    }

    fn version(&self) -> Option<&str> {
        self.inner.version()
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(a: u64, b: u64) -> Range<u64> {
        a..b
    }

    #[test]
    fn merge_adjacent_and_gapped() {
        // Adjacent and within-gap ranges merge, far ones stay separate.
        let m = merge_ranges(&[r(100, 200), r(200, 300), r(310, 400), r(10_000, 10_100)], 16, 1 << 20);
        assert_eq!(m, vec![r(100, 400), r(10_000, 10_100)]);
        // Zero gap merges only touching/overlapping ranges.
        let m = merge_ranges(&[r(0, 10), r(10, 20), r(21, 30)], 0, 1 << 20);
        assert_eq!(m, vec![r(0, 20), r(21, 30)]);
    }

    #[test]
    fn merge_respects_max_len_and_order() {
        let m = merge_ranges(&[r(300, 400), r(0, 100), r(100, 200), r(200, 300)], 0, 250);
        assert_eq!(m, vec![r(0, 200), r(200, 400)]);
        // One oversized range is kept whole.
        assert_eq!(merge_ranges(&[r(0, 1000)], 0, 100), vec![r(0, 1000)]);
    }

    #[test]
    fn merge_handles_duplicates_overlaps_and_empty() {
        let m = merge_ranges(&[r(5, 5), r(0, 50), r(0, 50), r(40, 60)], 0, 1 << 20);
        assert_eq!(m, vec![r(0, 60)]);
        assert!(merge_ranges(&[], 0, 10).is_empty());
    }
}

#[cfg(test)]
mod io_tests {
    use super::*;
    use crate::range_reader::MemoryRangeReader;
    use crate::test_support::MockReader;

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    fn expect(d: &[u8], r: &Range<u64>) -> Vec<u8> {
        d[r.start as usize..r.end as usize].to_vec()
    }

    #[tokio::test(start_paused = true)]
    async fn fetch_ranges_returns_requested_slices_in_order() {
        let d = data(100_000);
        let mock = MockReader::new(d.clone(), "mock://a", Duration::from_millis(10));
        let ranges = vec![50_000..50_100, 10..20, 10_000..10_050, 15..30, 0..0];
        let out = fetch_ranges(&mock, &ranges, &IoOptions { coalesce_gap: 100, ..IoOptions::default() }).await.unwrap();
        for (r, b) in ranges.iter().zip(&out) {
            assert_eq!(b.as_ref(), expect(&d, r).as_slice(), "{r:?}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn adjacent_ranges_become_one_request() {
        let d = data(100_000);
        let mock = MockReader::new(d, "mock://b", Duration::from_millis(10));
        let opts = IoOptions { coalesce_gap: 0, ..IoOptions::default() };
        // Four touching tiles and one far away: 2 requests.
        let ranges = vec![1000..1500, 1500..2200, 2200..2300, 2300..3000, 90_000..90_500];
        fetch_ranges(&mock, &ranges, &opts).await.unwrap();
        let mut calls = mock.calls();
        calls.sort_by_key(|c| c.start);
        assert_eq!(calls, vec![1000..3000, 90_000..90_500]);

        // A gap smaller than the threshold is bridged, a larger one is not.
        mock.reset();
        let opts = IoOptions { coalesce_gap: 200, ..IoOptions::default() };
        fetch_ranges(&mock, &[0..100, 250..300, 600..700], &opts).await.unwrap();
        let mut calls = mock.calls();
        calls.sort_by_key(|c| c.start);
        assert_eq!(calls, vec![0..300, 600..700]);
    }

    #[tokio::test(start_paused = true)]
    async fn requests_run_concurrently_up_to_the_limit() {
        let d = data(1_000_000);
        let latency = Duration::from_millis(100);
        let ranges: Vec<Range<u64>> = (0..8).map(|i| i * 100_000..i * 100_000 + 1000).collect();

        let mock = MockReader::new(d.clone(), "mock://c", latency);
        let opts = IoOptions { coalesce_gap: 0, max_concurrent_requests: 16, ..IoOptions::default() };
        let t0 = tokio::time::Instant::now();
        fetch_ranges(&mock, &ranges, &opts).await.unwrap();
        assert_eq!(t0.elapsed(), latency, "8 requests overlap");
        assert_eq!(mock.max_in_flight(), 8);

        let mock = MockReader::new(d, "mock://c2", latency);
        let opts = IoOptions { coalesce_gap: 0, max_concurrent_requests: 3, ..IoOptions::default() };
        let t0 = tokio::time::Instant::now();
        fetch_ranges(&mock, &ranges, &opts).await.unwrap();
        assert_eq!(mock.max_in_flight(), 3);
        assert_eq!(t0.elapsed(), latency * 3, "8 requests at 3 at a time take 3 rounds");
    }

    #[tokio::test(start_paused = true)]
    async fn first_error_fails_the_fetch() {
        let mock = MockReader::new(data(100_000), "mock://d", Duration::from_millis(10));
        mock.fail_on(50_000..50_001);
        let opts = IoOptions { coalesce_gap: 0, ..IoOptions::default() };
        let err = fetch_ranges(&mock, &[0..100, 50_000..50_100, 90_000..90_100], &opts).await.unwrap_err();
        assert!(err.to_string().contains("injected failure"), "{err}");
    }

    #[tokio::test(start_paused = true)]
    async fn short_response_is_an_error() {
        // Memory readers clamp reads at EOF; a fetch must not silently accept that.
        struct Clamping(MemoryRangeReader);
        impl AsyncRangeReader for Clamping {
            fn read_range(&self, offset: u64, len: usize) -> BoxFuture<'_, AnyResult<Bytes>> {
                Box::pin(async move { Ok(Bytes::from(self.0.read_range(offset, len)?)) })
            }
            fn size(&self) -> u64 {
                self.0.size()
            }
            fn identifier(&self) -> &str {
                self.0.identifier()
            }
        }
        let r = Clamping(MemoryRangeReader::new(data(100), "mem://short".into()));
        let range = 90..120;
        let err = fetch_ranges(&r, std::slice::from_ref(&range), &IoOptions::default()).await.unwrap_err();
        assert!(err.to_string().contains("short read"), "{err}");
    }

    #[test]
    fn sync_to_async_inline_needs_no_runtime() {
        let mem = MemoryRangeReader::new(data(1000), "mem://x".into());
        let r = SyncToAsync::new(Arc::new(mem));
        let b = futures::executor::block_on(r.read_range(10, 5)).unwrap();
        assert_eq!(b.as_ref(), &data(1000)[10..15]);
        let many = futures::executor::block_on(r.read_ranges(&[0..4, 4..8])).unwrap();
        assert_eq!(many.len(), 2);
    }

    #[tokio::test]
    async fn sync_to_async_runs_blocking_readers_on_the_blocking_pool() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, data(5000)).unwrap();
        let local = crate::range_reader::LocalRangeReader::new(&path).unwrap();
        let r = SyncToAsync::new(Arc::new(local));
        let b = r.read_range(100, 50).await.unwrap();
        assert_eq!(b.as_ref(), &data(5000)[100..150]);
    }

    fn remote_like() -> Arc<dyn AsyncRangeReader> {
        // Real (not paused) latency: the sync adapter drives this on the private runtime.
        Arc::new(MockReader::new(data(10_000), "mock://sync", Duration::from_millis(20)))
    }

    #[test]
    fn async_to_sync_from_a_plain_thread() {
        let r = AsyncToSync::new(remote_like());
        assert_eq!(r.read_range(100, 10).unwrap(), data(10_000)[100..110].to_vec());
        assert_eq!((r.size(), r.identifier(), r.is_local()), (10_000, "mock://sync", false));
    }

    #[tokio::test]
    async fn async_to_sync_from_a_blocking_thread() {
        let r = AsyncToSync::new(remote_like());
        let v = tokio::task::spawn_blocking(move || r.read_range(5, 5).unwrap()).await.unwrap();
        assert_eq!(v, data(10_000)[5..10].to_vec());
    }

    /// Calling the sync adapter from inside a runtime worker blocks that worker but must
    /// neither panic nor deadlock, even on a single-threaded runtime.
    #[tokio::test(flavor = "current_thread")]
    async fn async_to_sync_from_a_current_thread_runtime() {
        let r = AsyncToSync::new(remote_like());
        let (tx, rx) = std::sync::mpsc::channel();
        // Guard against a hang turning into a stuck test run.
        let watchdog = std::thread::spawn(move || {
            if rx.recv_timeout(Duration::from_secs(20)).is_err() {
                eprintln!("async_to_sync_from_a_current_thread_runtime deadlocked");
                std::process::abort();
            }
        });
        let v = r.read_range(0, 8).unwrap();
        assert_eq!(v, data(10_000)[0..8].to_vec());
        tx.send(()).unwrap();
        watchdog.join().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_to_sync_from_a_multi_thread_runtime_worker() {
        let r = AsyncToSync::new(remote_like());
        assert_eq!(r.read_range(0, 8).unwrap(), data(10_000)[0..8].to_vec());
    }
}
