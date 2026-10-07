//! Tests of the async extraction pipeline (plan -> fetch -> render) over a mock remote reader
//! that records the requests it receives. No network is involved.

use std::sync::Arc;
use std::time::Duration;

use crate::test_support::{build_cog, CogSpec, MockReader, Sample};
use crate::{BoundingBox, CogReader, IoOptions, MemoryRangeReader, OverviewQualityHint, ResamplingMethod, TileData, TileExtractor};

/// 512x512 u8 COG with 64x64 tiles (8x8 = 64 tiles), covering XYZ tile 3/2/3.
fn spec() -> CogSpec {
    let ext = BoundingBox::from_xyz(3, 2, 3);
    let px = (ext.maxx - ext.minx) / 512.0;
    CogSpec {
        width: 512,
        height: 512,
        tile: 64,
        bands: 1,
        sample: Sample::U8,
        deflate: true,
        predictor: false,
        epsg: 3857,
        origin: (ext.minx, ext.maxy),
        pixel_size: (px, px),
        nodata: None,
        overviews: 0,
        sparse: vec![],
        corrupt: vec![],
        pixel: |_, x, y| ((x * 5 + y * 3) % 250) as f64 + 1.0,
    }
}

async fn open_mock(id: &str, latency: Duration, options: IoOptions) -> (Arc<MockReader>, CogReader) {
    let mock = Arc::new(MockReader::new(build_cog(&spec()), id, latency).with_options(options));
    let reader = CogReader::from_async_reader_with_hint(mock.clone(), OverviewQualityHint::NoneUsable)
        .await
        .unwrap();
    mock.reset();
    (mock, reader)
}

fn no_coalescing() -> IoOptions {
    IoOptions { coalesce_gap: 0, ..IoOptions::default() }
}

async fn extract(reader: &CogReader, z: u32, x: u32, y: u32) -> TileData {
    TileExtractor::new(reader).xyz(z, x, y).size(128).extract().await.unwrap()
}

/// Reference result from a plain in-memory reader (its own cache identity).
async fn reference(id: &str, z: u32, x: u32, y: u32) -> TileData {
    let r = CogReader::from_reader_with_hint(
        Arc::new(MemoryRangeReader::new(build_cog(&spec()), id.to_string())),
        OverviewQualityHint::NoneUsable,
    )
    .unwrap();
    extract(&r, z, x, y).await
}

#[tokio::test(start_paused = true)]
async fn adjacent_source_tiles_are_fetched_in_one_request() {
    let (mock, reader) = open_mock("mock://x/coalesce", Duration::from_millis(50), IoOptions::default()).await;
    let tile = extract(&reader, 3, 2, 3).await;
    assert!(tile.tiles_read > 8, "expected a multi-tile extraction, got {}", tile.tiles_read);
    // The tiles sit back to back in the file, so one request covers all of them.
    assert_eq!(mock.call_count(), 1, "{:?}", mock.calls());
}

#[tokio::test(start_paused = true)]
async fn without_coalescing_each_row_of_tiles_is_one_request_and_they_overlap_in_time() {
    let latency = Duration::from_millis(100);
    let (mock, reader) = open_mock("mock://x/rows", latency, no_coalescing()).await;
    let t0 = tokio::time::Instant::now();
    let tile = extract(&reader, 4, 4, 6).await;
    let elapsed = t0.elapsed();

    let calls = mock.calls();
    assert!(calls.len() > 1 && calls.len() < tile.tiles_read, "{} requests for {} tiles", calls.len(), tile.tiles_read);
    // No bytes beyond the tiles themselves were fetched.
    assert_eq!(calls.iter().map(|c| (c.end - c.start) as usize).sum::<usize>(), tile.bytes_fetched);
    // All requests were in flight together: total time is one round trip, not one per request.
    assert_eq!(mock.max_in_flight(), calls.len());
    assert!(elapsed < latency * 2, "took {elapsed:?} for {} requests", calls.len());
}

#[tokio::test(start_paused = true)]
async fn concurrency_per_extraction_is_capped() {
    let latency = Duration::from_millis(100);
    let options = IoOptions { coalesce_gap: 0, max_concurrent_requests: 2, ..IoOptions::default() };
    let (mock, reader) = open_mock("mock://x/capped", latency, options).await;
    extract(&reader, 4, 4, 6).await;
    assert!(mock.call_count() > 2);
    assert_eq!(mock.max_in_flight(), 2);
}

#[tokio::test(start_paused = true)]
async fn a_cached_tile_costs_no_request() {
    let (mock, reader) = open_mock("mock://x/cached", Duration::from_millis(10), IoOptions::default()).await;
    let first = extract(&reader, 3, 2, 3).await;
    assert!(mock.call_count() > 0);
    mock.reset();
    let second = extract(&reader, 3, 2, 3).await;
    assert_eq!(mock.call_count(), 0);
    assert_eq!((second.bytes_fetched, second.tiles_read), (0, 0));
    assert_eq!(first.pixels, second.pixels);
}

#[tokio::test(start_paused = true)]
async fn concurrent_extractions_of_the_same_tiles_share_one_fetch() {
    let (mock, reader) = open_mock("mock://x/singleflight", Duration::from_millis(50), IoOptions::default()).await;
    let other = reader.clone(); // same source identity, as after a repeated open
    let (a, b, c) = tokio::join!(
        extract(&reader, 3, 2, 3),
        extract(&other, 3, 2, 3),
        extract(&reader, 3, 2, 3)
    );
    assert_eq!(mock.call_count(), 1, "{:?}", mock.calls());
    assert_eq!(a.pixels, b.pixels);
    assert_eq!(a.pixels, c.pixels);
    // Exactly one of them fetched; the others report no I/O of their own.
    let fetched: Vec<usize> = [&a, &b, &c].iter().map(|t| t.tiles_read).collect();
    assert_eq!(fetched.iter().filter(|n| **n > 0).count(), 1, "{fetched:?}");
    assert_eq!(crate::tile_fetch::inflight_count(reader.identifier()), 0, "in-flight table must drain");
}

#[tokio::test(start_paused = true)]
async fn overlapping_extractions_fetch_each_source_tile_once() {
    let (mock, reader) = open_mock("mock://x/overlap", Duration::from_millis(50), no_coalescing()).await;
    // The two z4 tiles on the top row of the COG share no source tiles; the parent shares all.
    let (a, b, parent) = tokio::join!(extract(&reader, 4, 4, 6), extract(&reader, 4, 5, 6), extract(&reader, 3, 2, 3));
    let distinct_bytes: usize = a.bytes_fetched + b.bytes_fetched + parent.bytes_fetched;
    let requested: usize = mock.calls().iter().map(|c| (c.end - c.start) as usize).sum();
    assert_eq!(requested, distinct_bytes, "no tile may be requested twice");
}

#[tokio::test(start_paused = true)]
async fn a_dropped_leader_hands_its_tiles_to_a_waiting_extraction() {
    let latency = Duration::from_millis(100);
    let (mock, reader) = open_mock("mock://x/cancel", latency, IoOptions::default()).await;
    let expected = reference("mem://cancel-ref", 3, 2, 3).await;

    let leader = tokio::time::timeout(latency / 2, extract(&reader, 3, 2, 3));
    let (leader_result, follower) = tokio::join!(leader, extract(&reader, 3, 2, 3));
    assert!(leader_result.is_err(), "the leader should have been cancelled mid-fetch");
    assert_eq!(follower.pixels, expected.pixels);
    // The first request was abandoned, the follower issued its own.
    assert_eq!(mock.call_count(), 2);
    assert_eq!(crate::tile_fetch::inflight_count(reader.identifier()), 0);
}

#[tokio::test(start_paused = true)]
async fn read_errors_fail_the_extraction_and_its_followers() {
    let (mock, reader) = open_mock("mock://x/error", Duration::from_millis(20), IoOptions::default()).await;
    let spans: Vec<_> = reader.metadata.tile_offsets.iter().zip(&reader.metadata.tile_byte_counts).collect();
    let (off, len) = spans[20];
    mock.fail_on(*off..off + len);

    let (a, b) = tokio::join!(
        TileExtractor::new(&reader).xyz(3, 2, 3).size(64).extract(),
        TileExtractor::new(&reader).xyz(3, 2, 3).size(64).extract()
    );
    for r in [a, b] {
        let err = r.unwrap_err().to_string();
        assert!(err.starts_with("Failed to read source tile "), "{err}");
        assert!(err.contains("(overview None)") && err.contains("injected failure"), "{err}");
    }
    assert_eq!(crate::tile_fetch::inflight_count(reader.identifier()), 0);

    // A later extraction that avoids the bad tile still works and the failure was not cached.
    mock.reset();
    let bounds = BoundingBox::from_xyz(5, 11, 15); // bottom-right corner of the COG
    let tile = TileExtractor::new(&reader).bounds(bounds).size(32).extract().await;
    assert!(tile.is_ok(), "{tile:?}");
}

#[tokio::test(start_paused = true)]
async fn output_matches_the_synchronous_reader_exactly() {
    let (_mock, remote) = open_mock("mock://x/parity", Duration::from_millis(5), no_coalescing()).await;
    for (z, x, y) in [(3, 2, 3), (4, 4, 6), (4, 5, 7), (5, 8, 12), (2, 1, 1), (6, 17, 25)] {
        for (name, method) in
            [("nearest", ResamplingMethod::Nearest), ("bilinear", ResamplingMethod::Bilinear), ("bicubic", ResamplingMethod::Bicubic)]
        {
            let got = TileExtractor::new(&remote).xyz(z, x, y).size(128).resampling(method).extract().await.unwrap();
            let r = CogReader::from_reader_with_hint(
                Arc::new(MemoryRangeReader::new(build_cog(&spec()), format!("mem://parity/{z}/{x}/{y}/{name}"))),
                OverviewQualityHint::NoneUsable,
            )
            .unwrap();
            let want = TileExtractor::new(&r).xyz(z, x, y).size(128).resampling(method).extract().await.unwrap();
            assert_eq!(got.pixels.len(), want.pixels.len());
            assert!(
                got.pixels.iter().zip(&want.pixels).all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
                "{z}/{x}/{y} {name}"
            );
            assert_eq!((got.overview_used, got.width, got.height, got.bands), (want.overview_used, want.width, want.height, want.bands));
        }
    }
}

/// With a tiny blocking pool, many concurrent remote extractions must not queue behind it: the
/// network waits happen on async tasks, and the blocking pool only sees short CPU jobs.
#[test]
fn remote_extractions_do_not_occupy_blocking_threads_while_waiting() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let latency = Duration::from_millis(300);
    let elapsed = rt.block_on(async {
        let mut opened = Vec::new();
        for i in 0..12 {
            let (mock, reader) = open_mock(&format!("mock://x/starve/{i}"), Duration::ZERO, no_coalescing()).await;
            mock.set_latency(latency);
            opened.push(reader);
        }
        let start = std::time::Instant::now();
        let results = futures::future::join_all(opened.iter().map(|r| extract(r, 4, 4, 6))).await;
        assert!(results.iter().all(|t| t.tiles_read > 0));
        start.elapsed()
    });
    // If each extraction held the only blocking thread for its 300 ms round trip this would
    // take 12 * 300 ms = 3.6 s.
    assert!(elapsed < Duration::from_millis(2000), "took {elapsed:?}");
}

#[tokio::test]
async fn extraction_and_open_futures_are_send() {
    fn assert_send<T: Send>(_: &T) {}
    let (_mock, reader) = open_mock("mock://x/send", Duration::ZERO, IoOptions::default()).await;
    let fut = TileExtractor::new(&reader).xyz(3, 2, 3).extract();
    assert_send(&fut);
    drop(fut);
    let open = CogReader::open_async("/nonexistent.tif");
    assert_send(&open);
    drop(open);
}

#[tokio::test(start_paused = true)]
async fn concurrent_opens_share_the_overview_quality_sampling() {
    // Several requests opening the same COG at once (default hint: sample overview tiles)
    // fetch each sample tile once between them.
    let mut spec = spec();
    spec.overviews = 2;
    let mock = Arc::new(MockReader::new(build_cog(&spec), "mock://x/open-share", Duration::from_millis(20)));
    let opens = (0..8).map(|_| CogReader::from_async_reader(mock.clone()));
    let readers = futures::future::join_all(opens).await;
    let first = readers[0].as_ref().unwrap();
    assert!(readers.iter().all(|r| r.as_ref().unwrap().min_usable_overview == first.min_usable_overview));

    let sampled_tile_requests = mock.calls().into_iter().filter(|c| c.start > 16 * 1024).count();
    let overview_tiles = first.overviews.last().unwrap().tile_offsets.len().min(3);
    assert!(
        sampled_tile_requests <= overview_tiles,
        "{sampled_tile_requests} tile requests for 8 opens, expected at most {overview_tiles}: {:?}",
        mock.calls()
    );
}
