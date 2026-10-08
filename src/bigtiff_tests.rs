//! BigTIFF (TIFF version 43) COGs: parsed the same as the classic-TIFF encoding of the same data.
//!
//! The synthetic cases come from the test encoder in both layouts (BigTIFF with TileOffsets /
//! TileByteCounts as LONG and as LONG8); the GDAL cases from `gdal_translate -of COG
//! -co BIGTIFF=YES|NO` (skipped when `gdal_translate` or a fixture is missing).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use crate::async_io::SyncToAsync;
use crate::cog_reader::{CogReader, OverviewQualityHint as Hint, TileRef};
use crate::range_reader::MemoryRangeReader;
use crate::test_support::{build_bigtiff_cog, build_cog, CogSpec, ObjectServer, Sample, ServedObject};
use crate::{BoundingBox, CacheConfig, CogCache, IoOptions, TileExtractor};

fn spec(width: usize, height: usize, tile: usize, bands: usize, sample: Sample) -> CogSpec {
    CogSpec {
        width,
        height,
        tile,
        bands,
        sample,
        deflate: true,
        predictor: false,
        epsg: 3857,
        origin: (0.0, 6400.0),
        pixel_size: (10.0, 10.0),
        nodata: None,
        overviews: 0,
        sparse: vec![],
        corrupt: vec![],
        pixel: |b, x, y| ((x * 7 + y * 13 + b * 50) % 250) as f64 + 1.0,
    }
}

/// Cases covering inline and out-of-line values in both layouts: BitsPerSample of 3 bands (6
/// bytes: inline only in BigTIFF), one-tile and two-tile arrays (inline in BigTIFF), an ASCII
/// nodata that fits 8 bytes but not 4, sparse tiles, overviews, predictors and edge tiles.
fn cases() -> Vec<(&'static str, CogSpec)> {
    let mut rgb = spec(300, 200, 64, 3, Sample::U8);
    rgb.predictor = true;
    rgb.overviews = 1;
    rgb.sparse = vec![(0, 3)];
    rgb.nodata = Some(0.0);
    rgb.epsg = 4326;
    rgb.origin = (10.0, 50.0);
    rgb.pixel_size = (0.01, 0.01);

    let mut float = spec(256, 192, 64, 1, Sample::F32);
    float.deflate = false;
    float.overviews = 2;
    float.nodata = Some(-9999.0);

    let one_tile = spec(64, 64, 64, 1, Sample::U16);
    let two_tiles = spec(128, 64, 64, 1, Sample::U16);

    let mut edges = spec(130, 70, 64, 3, Sample::U8);
    edges.overviews = 1;

    vec![("rgb", rgb), ("float", float), ("one_tile", one_tile), ("two_tiles", two_tiles), ("edges", edges)]
}

fn open_sync(bytes: Vec<u8>, id: &str) -> CogReader {
    CogReader::from_reader_with_hint(Arc::new(MemoryRangeReader::new(bytes, id.to_string())), Hint::NoneUsable).unwrap()
}

async fn open_async(bytes: Vec<u8>, id: &str) -> CogReader {
    let io = SyncToAsync::new(Arc::new(MemoryRangeReader::new(bytes, id.to_string())));
    CogReader::from_async_reader_with_hint(Arc::new(io), Hint::NoneUsable).await.unwrap()
}

fn same_pixels(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// Same header, geometry, tile layout and decoded pixels; only the byte offsets may differ.
fn assert_same(classic: &CogReader, big: &CogReader, what: &str) {
    let (a, b) = (&*classic.metadata, &*big.metadata);
    assert_eq!((a.width, a.height, a.tile_width, a.tile_height), (b.width, b.height, b.tile_width, b.tile_height), "{what}");
    assert_eq!((a.bands, a.data_type, a.compression, a.predictor), (b.bands, b.data_type, b.compression, b.predictor), "{what}");
    assert_eq!((a.tiles_across, a.tiles_down, a.is_tiled, a.little_endian), (b.tiles_across, b.tiles_down, b.is_tiled, b.little_endian), "{what}");
    assert_eq!((a.crs_code, a.nodata, a.stats_min, a.stats_max), (b.crs_code, b.nodata, b.stats_min, b.stats_max), "{what}");
    assert_eq!(format!("{:?}", a.geo_transform), format!("{:?}", b.geo_transform), "{what}");
    assert_eq!(a.tile_byte_counts, b.tile_byte_counts, "{what}");
    assert_eq!(classic.overviews.len(), big.overviews.len(), "{what}");

    let levels: Vec<(Option<usize>, usize)> = std::iter::once((None, a.tile_offsets.len()))
        .chain(classic.overviews.iter().enumerate().map(|(i, o)| (Some(i), o.tile_offsets.len())))
        .collect();
    for (overview, tiles) in levels {
        if let Some(i) = overview {
            let (x, y) = (&classic.overviews[i], &big.overviews[i]);
            assert_eq!((x.width, x.height, x.tile_width, x.scale), (y.width, y.height, y.tile_width, y.scale), "{what}");
            assert_eq!((x.scale_x, x.scale_y), (y.scale_x, y.scale_y), "{what}");
            assert_eq!(x.tile_byte_counts, y.tile_byte_counts, "{what}");
        }
        for index in 0..tiles {
            let tile = TileRef { overview, index };
            let (c, _) = classic.read_tile_sync(tile).unwrap();
            let (g, _) = big.read_tile_sync(tile).unwrap();
            assert!(same_pixels(&c, &g), "{what}: tile {index} of {overview:?} differs");
        }
    }
}

#[test]
fn bigtiff_parses_like_classic_in_both_array_types() {
    for (name, spec) in cases() {
        let classic = open_sync(build_cog(&spec), &format!("mem://bigtiff/{name}/classic"));
        for long8 in [true, false] {
            let bytes = build_bigtiff_cog(&spec, long8);
            assert_eq!(&bytes[..4], b"II+\0", "the fixture is a BigTIFF");
            let big = open_sync(bytes, &format!("mem://bigtiff/{name}/big-{long8}"));
            assert_same(&classic, &big, &format!("{name} long8={long8}"));
            if name == "float" {
                assert_eq!(big.metadata.nodata, Some(-9999.0), "ASCII nodata stored inline in the IFD entry");
            }
        }
    }
}

/// BitsPerSample and SampleFormat hold one value per band: three of them are out of line in a
/// classic TIFF and inline in a BigTIFF. Both must give the data type and the pixels.
#[test]
fn multi_band_wide_samples_keep_their_data_type() {
    for (sample, data_type) in [(Sample::U16, crate::CogDataType::UInt16), (Sample::F32, crate::CogDataType::Float32)] {
        let spec = spec(64, 64, 64, 3, sample);
        for (layout, bytes) in [("classic", build_cog(&spec)), ("big", build_bigtiff_cog(&spec, true))] {
            let what = format!("{data_type:?} {layout}");
            let reader = open_sync(bytes, &format!("mem://bigtiff/multi-band/{what}"));
            assert_eq!(reader.metadata.data_type, data_type, "{what}");
            let (tile, _) = reader.read_tile_sync(TileRef { overview: None, index: 0 }).unwrap();
            for (i, &v) in tile.iter().enumerate() {
                let (b, x, y) = (i % 3, i / 3 % 64, i / 3 / 64);
                assert_eq!(f64::from(v), (spec.pixel)(b, x, y), "{what}: band {b} pixel ({x}, {y})");
            }
        }
    }
}

#[tokio::test]
async fn bigtiff_parses_like_classic_on_the_async_path() {
    for (name, spec) in cases() {
        let classic = open_async(build_cog(&spec), &format!("mem://bigtiff-async/{name}/classic")).await;
        for long8 in [true, false] {
            let big = open_async(build_bigtiff_cog(&spec, long8), &format!("mem://bigtiff-async/{name}/big-{long8}")).await;
            assert_same(&classic, &big, &format!("{name} long8={long8} (async)"));
        }
    }
}

#[tokio::test]
async fn bigtiff_extraction_and_point_queries_match_classic() {
    use crate::{PointQuery, ResamplingMethod};
    for (name, spec) in cases() {
        let (w, h) = (spec.width as f64 * spec.pixel_size.0, spec.height as f64 * spec.pixel_size.1);
        let (minx, maxy) = spec.origin;
        let bounds = BoundingBox::new(minx + 0.1 * w, maxy - 0.9 * h, minx + 0.8 * w, maxy - 0.15 * h);
        let classic = open_async(build_cog(&spec), &format!("mem://bigtiff-x/{name}/classic")).await;
        let big = open_async(build_bigtiff_cog(&spec, true), &format!("mem://bigtiff-x/{name}/big")).await;
        for method in [ResamplingMethod::Nearest, ResamplingMethod::Bilinear] {
            let extract = |r: CogReader| async move {
                // Output in the raster's own CRS so the windows line up
                TileExtractor::new(&r).bounds(bounds).output_crs(u32::from(spec.epsg)).size(48).resampling(method).extract().await
            };
            let (c, g) = (extract(classic.clone()).await.unwrap(), extract(big.clone()).await.unwrap());
            assert!(same_pixels(&c.pixels, &g.pixels), "{name} {method:?}");
            assert_eq!(c.tiles_read, g.tiles_read, "{name} {method:?}");
            assert!(c.pixels.iter().any(|v| !v.is_nan()), "{name}: the window must cover data");
        }
        // Point queries (sync and async) through the BigTIFF reader
        let (px, py) = (minx + 0.4 * w, maxy - 0.5 * h);
        let crs = i32::from(spec.epsg);
        let a = classic.sample_crs_async(crs, px, py).await.unwrap();
        let b = big.sample_crs_async(crs, px, py).await.unwrap();
        assert_eq!(a.values.iter().map(|(k, v)| (*k, v.to_bits())).collect::<std::collections::BTreeMap<_, _>>(),
                   b.values.iter().map(|(k, v)| (*k, v.to_bits())).collect::<std::collections::BTreeMap<_, _>>(), "{name}");
        let sync = tokio::task::spawn_blocking({
            let big = big.clone();
            move || big.sample_crs(crs, px, py)
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(sync.values.len(), b.values.len());
    }
}

fn http_open(server: &ObjectServer, path: &str) -> impl std::future::Future<Output = CogReader> {
    let url = format!("{}{path}", server.base());
    async move {
        let options = IoOptions { max_retries: 0, ..IoOptions::default() };
        CogReader::builder(&url)
            .cache(&CogCache::disabled())
            .hint(Hint::AllUsable)
            .io_options(options)
            .open_async()
            .await
            .unwrap()
    }
}

#[tokio::test]
async fn bigtiff_over_http_opens_in_as_many_requests_as_classic_and_reads_tiles() {
    let mut s = spec(512, 512, 64, 3, Sample::U8);
    s.overviews = 2;
    let classic = ObjectServer::start(Some(ServedObject::new(build_cog(&s))), Duration::ZERO);
    let big = ObjectServer::start(Some(ServedObject::new(build_bigtiff_cog(&s, true))), Duration::ZERO);

    let (c, b) = (http_open(&classic, "/c.tif").await, http_open(&big, "/b.tif").await);
    assert_eq!(big.requests().len(), classic.requests().len(), "same request count at open");
    assert_eq!(big.requests().len(), 1, "the 16 KiB prefix holds the header and every IFD");
    assert_same(&open_sync(build_cog(&s), "mem://bigtiff-http/classic"), &b, "over http");

    // A tile read is a conditional range request for the BigTIFF's version, like the classic one
    let (tile_c, tile_b) = (c.read_tile_async(5).await.unwrap(), b.read_tile_async(5).await.unwrap());
    assert!(same_pixels(&tile_c, &tile_b));
    let read = big.requests().pop().unwrap();
    assert_eq!(read.if_match.as_deref(), Some("\"abc\""));
}

#[tokio::test]
async fn bigtiff_headers_are_cached_like_classic_ones() {
    let mut s = spec(512, 512, 64, 3, Sample::U8);
    s.overviews = 2;
    let tiles: usize = (0..=s.overviews).map(|l| s.tiles_across(l) * s.tiles_down(l)).sum();
    let big = ObjectServer::start(Some(ServedObject::new(build_bigtiff_cog(&s, true))), Duration::ZERO);
    let url = format!("{}/cached.tif", big.base());
    let cache = CogCache::new(CacheConfig::default());
    let open = || CogReader::builder(&url).cache(&cache).hint(Hint::ComputeAtRuntime).open_async();

    let first = open().await.unwrap();
    let after_first = big.requests().len();
    let second = open().await.unwrap();
    assert_eq!(big.requests().len(), after_first, "a hit sends no request");
    assert!(Arc::ptr_eq(&first.metadata, &second.metadata));
    // 16 bytes per tile (u64 offset and byte count, whatever width the file stored them in)
    let weight = cache.stats().headers.bytes;
    assert!(weight >= 16 * tiles && weight < 16 * tiles + 8 * 1024, "weight {weight} for {tiles} tiles");
    assert!(first.read_tile_async(3).await.is_ok());
}

#[test]
fn local_bigtiff_files_open_through_every_entry_point() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = spec(200, 130, 64, 1, Sample::F32);
    s.deflate = false;
    s.overviews = 1;
    let (classic_path, big_path) = (dir.path().join("classic.tif"), dir.path().join("big.tif"));
    std::fs::write(&classic_path, build_cog(&s)).unwrap();
    std::fs::write(&big_path, build_bigtiff_cog(&s, true)).unwrap();
    let (cp, bp) = (classic_path.to_str().unwrap(), big_path.to_str().unwrap());

    let classic = CogReader::open_with_hint(cp, Hint::NoneUsable).unwrap();
    let big = CogReader::open_with_hint(bp, Hint::NoneUsable).unwrap();
    assert_same(&classic, &big, "local sync open");
    let big_default = CogReader::open(bp).unwrap();
    assert_eq!(big_default.metadata.width, 200);

    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let async_big = runtime.block_on(CogReader::open_async(bp)).unwrap();
    assert_same(&classic, &async_big, "local async open");
}

#[test]
fn the_chunked_and_lzw_file_readers_say_clearly_that_they_do_not_read_bigtiff() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.tif");
    std::fs::write(&path, build_bigtiff_cog(&spec(64, 64, 64, 1, Sample::U8), false)).unwrap();

    let mut file = std::fs::File::open(&path).unwrap();
    let err = crate::tiff_utils::read_tiff_header(&mut file).unwrap_err().to_string();
    assert!(err.contains("BigTIFF") && err.contains("CogReader"), "{err}");
    let err = crate::tiff_utils::read_primary_compression(&path).unwrap_err().to_string();
    assert!(err.contains("BigTIFF"), "{err}");
}

// ---- GDAL-made COGs ----

const NATURAL_EARTH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/natural_earth_rgb.tif");
const DEM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/copernicus_dem_san_francisco.tif");

/// `gdal_translate -of COG` of `src` into `dir`, as BigTIFF or classic TIFF; `None` if
/// `gdal_translate` or `src` is not available.
fn gdal_cog(src: &str, dir: &Path, name: &str, big: bool, extra: &[&str]) -> Option<PathBuf> {
    if !Path::new(src).exists() {
        println!("Skipping: {src} not found");
        return None;
    }
    let out = dir.join(name);
    let status = Command::new("gdal_translate")
        .args(["-q", "-of", "COG", "-co"])
        .arg(format!("BIGTIFF={}", if big { "YES" } else { "NO" }))
        .args(extra)
        .arg(src)
        .arg(&out)
        .status()
        .ok()?;
    status.success().then_some(out)
}

fn version_of(path: &Path) -> u8 {
    std::fs::read(path).unwrap()[2]
}

async fn compare_gdal_pair(src: &str, extra: &[&str], what: &str) {
    let dir = tempfile::tempdir().unwrap();
    let Some(classic) = gdal_cog(src, dir.path(), "classic.tif", false, extra) else { return };
    let big = gdal_cog(src, dir.path(), "big.tif", true, extra).expect("same tool, same inputs");
    assert_eq!((version_of(&classic), version_of(&big)), (42, 43), "{what}: GDAL wrote the layouts asked for");

    let (cp, bp) = (classic.to_str().unwrap(), big.to_str().unwrap());
    let classic_reader = CogReader::open_async_with_hint(cp, Hint::NoneUsable).await.unwrap();
    let big_reader = CogReader::open_async_with_hint(bp, Hint::NoneUsable).await.unwrap();
    assert!(!big_reader.overviews.is_empty(), "{what}: the COG has overviews");
    assert_same(&classic_reader, &big_reader, what);

    // The BigTIFF through GDAL itself: the first block of band 1
    let ds = gdal::Dataset::open(&big).unwrap();
    let band = ds.rasterband(1).unwrap();
    let (tw, th) = (big_reader.metadata.tile_width, big_reader.metadata.tile_height);
    let (w, h) = (tw.min(big_reader.metadata.width), th.min(big_reader.metadata.height));
    let reference: gdal::raster::Buffer<f32> = band.read_as((0, 0), (w, h), (w, h), None).unwrap();
    let (tile, _) = big_reader.read_tile_sync(TileRef { overview: None, index: 0 }).unwrap();
    let bands = big_reader.metadata.bands;
    for y in 0..h {
        for x in 0..w {
            let ours = tile[(y * tw + x) * bands];
            let theirs = reference.data()[y * w + x];
            assert!(ours.to_bits() == theirs.to_bits() || (ours.is_nan() && theirs.is_nan()), "{what}: ({x},{y}) {ours} vs {theirs}");
        }
    }
}

#[tokio::test]
async fn gdal_bigtiff_cog_equals_the_classic_cog_rgb_deflate() {
    compare_gdal_pair(NATURAL_EARTH, &["-co", "COMPRESS=DEFLATE", "-co", "BLOCKSIZE=256"], "natural_earth deflate").await;
}

#[tokio::test]
async fn gdal_bigtiff_cog_equals_the_classic_cog_rgb_lzw() {
    compare_gdal_pair(NATURAL_EARTH, &["-co", "COMPRESS=LZW", "-co", "BLOCKSIZE=256"], "natural_earth lzw").await;
}

#[tokio::test]
async fn gdal_bigtiff_cog_equals_the_classic_cog_float_dem() {
    compare_gdal_pair(DEM, &["-co", "COMPRESS=DEFLATE", "-co", "BLOCKSIZE=512"], "copernicus dem (Float32, PixelIsPoint)").await;
}

#[tokio::test]
async fn gdal_bigtiff_cog_equals_the_classic_cog_zstd() {
    compare_gdal_pair(NATURAL_EARTH, &["-co", "COMPRESS=ZSTD", "-co", "BLOCKSIZE=512"], "natural_earth zstd").await;
}

/// A realistically sized COG (16000 x 8000 RGB, 512 px tiles, three overviews, ~6 MB of
/// compressed data; 2 threads so as not to starve timing-sensitive tests) in both layouts:
/// requests needed to open it over HTTP.
#[tokio::test]
async fn gdal_bigtiff_cog_opens_in_few_requests_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let extra = ["-co", "COMPRESS=DEFLATE", "-co", "BLOCKSIZE=512", "-co", "NUM_THREADS=2", "-outsize", "16000", "8000", "-r", "near"];
    let Some(classic) = gdal_cog(NATURAL_EARTH, dir.path(), "classic.tif", false, &extra) else { return };
    let big = gdal_cog(NATURAL_EARTH, dir.path(), "big.tif", true, &extra).unwrap();
    assert_eq!((version_of(&classic), version_of(&big)), (42, 43));

    let mut counts = Vec::new();
    for (label, path) in [("classic", &classic), ("bigtiff", &big)] {
        let server = ObjectServer::start(Some(ServedObject::new(std::fs::read(path).unwrap())), Duration::ZERO);
        let reader = http_open(&server, "/big.tif").await;
        let requests = server.requests();
        println!(
            "OPENREQ {label}: {} requests ({:?}), {} tiles, {} overviews",
            requests.len(),
            requests.iter().map(|r| r.range).collect::<Vec<_>>(),
            reader.metadata.tile_offsets.len() + reader.overviews.iter().map(|o| o.tile_offsets.len()).sum::<usize>(),
            reader.overviews.len(),
        );
        counts.push(requests.len());
        // and it reads: a tile deep in the file
        let last = reader.metadata.tile_offsets.len() - 1;
        assert!(reader.read_tile_async(last).await.is_ok());
    }
    assert!(counts[1] <= counts[0] + 1, "BigTIFF open costs {counts:?} requests");
}

/// Measurement (run with `--ignored --nocapture`): requests to open classic and BigTIFF COGs of
/// growing size over HTTP, and bytes fetched at open. The tile arrays outgrow the 16 KiB prefix
/// (BigTIFF's are twice as big, 8 bytes per entry) and each array past it is its own request.
#[tokio::test]
#[ignore = "measurement; generates COGs of up to 60000 x 30000 pixels"]
async fn gdal_open_request_counts_by_size() {
    let dir = tempfile::tempdir().unwrap();
    for (w, h) in [(16000, 8000), (30000, 15000), (60000, 30000)] {
        let (ws, hs) = (w.to_string(), h.to_string());
        let extra = ["-co", "COMPRESS=DEFLATE", "-co", "BLOCKSIZE=512", "-co", "NUM_THREADS=ALL_CPUS", "-outsize", &ws, &hs, "-r", "near"];
        let Some(classic) = gdal_cog(NATURAL_EARTH, dir.path(), &format!("c{w}.tif"), false, &extra) else { return };
        let big = gdal_cog(NATURAL_EARTH, dir.path(), &format!("b{w}.tif"), true, &extra).unwrap();
        for (label, path) in [("classic", &classic), ("bigtiff", &big)] {
            let server = ObjectServer::start(Some(ServedObject::new(std::fs::read(path).unwrap())), Duration::ZERO);
            let before = crate::io_stats();
            let reader = http_open(&server, "/m.tif").await;
            let io = crate::io_stats();
            let tiles = reader.metadata.tile_offsets.len() + reader.overviews.iter().map(|o| o.tile_offsets.len()).sum::<usize>();
            println!(
                "OPENSIZE {w}x{h} {label}: {} requests, {} bytes fetched, {tiles} tiles, file {} MB",
                io.requests - before.requests,
                io.bytes - before.bytes,
                std::fs::metadata(path).unwrap().len() / 1_000_000,
            );
        }
    }
}

/// ArcticDEM v4.1 2 m mosaic tile (PGC's public bucket, anonymous): a 166 MB BigTIFF COG,
/// 25100 x 25100 Float32, LZW with the floating-point predictor, 512 px tiles, six overviews.
const ARCTICDEM: &str =
    "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/2m/07_40/07_40_2_2_2m_v4.1_dem.tif";

/// Network smoke test: open a real BigTIFF COG over HTTPS and over `s3://`, read the tile with
/// the most data and compare it with GDAL's read of the same block through `/vsicurl/`.
#[tokio::test]
#[ignore = "needs network access"]
async fn real_public_bigtiff_cog_opens_and_reads_a_tile_like_gdal() {
    let before = crate::io_stats();
    let reader = CogReader::builder(ARCTICDEM).cache(&CogCache::disabled()).hint(Hint::AllUsable).open_async().await.unwrap();
    let open = crate::io_stats();
    println!(
        "ARCTICDEM open: {} requests, {} bytes; {}x{} {:?} {:?} predictor {}; {} tiles, {} overviews",
        open.requests - before.requests,
        open.bytes - before.bytes,
        reader.metadata.width,
        reader.metadata.height,
        reader.metadata.data_type,
        reader.metadata.compression,
        reader.metadata.predictor,
        reader.metadata.tile_offsets.len(),
        reader.overviews.len(),
    );
    let m = &reader.metadata;
    assert_eq!(reader.io().read_range(0, 4).await.unwrap()[..], b"II+\0"[..], "the object is a BigTIFF");
    assert_eq!((m.width, m.height, m.tile_width, m.tile_height), (25100, 25100, 512, 512));
    assert_eq!((m.crs_code, m.nodata, m.predictor), (Some(3413), Some(-9999.0), 3));
    assert_eq!(reader.overviews.len(), 6);
    assert!(m.tile_offsets.iter().all(|&o| o < reader.io().size()), "offsets are inside the file");

    // The tile with the most data, against GDAL
    let (index, _) = m.tile_byte_counts.iter().enumerate().max_by_key(|(_, c)| **c).unwrap();
    let (tx, ty) = (index % m.tiles_across, index / m.tiles_across);
    let ours = reader.read_tile_async(index).await.unwrap();
    let ds = gdal::Dataset::open(format!("/vsicurl/{ARCTICDEM}")).unwrap();
    let (x0, y0) = (tx * 512, ty * 512);
    let (w, h) = (512.min(m.width - x0), 512.min(m.height - y0));
    let theirs: gdal::raster::Buffer<f32> =
        ds.rasterband(1).unwrap().read_as((x0 as isize, y0 as isize), (w, h), (w, h), None).unwrap();
    let mut valid = 0usize;
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (ours[y * 512 + x], theirs.data()[y * w + x]);
            assert_eq!(a.to_bits(), b.to_bits(), "tile {index} ({tx},{ty}) pixel ({x},{y}): {a} vs {b}");
            valid += usize::from(a != -9999.0);
        }
    }
    println!("ARCTICDEM tile {index} ({tx},{ty}): {valid} of {} pixels have data, identical to GDAL", w * h);
    assert!(valid > 1000);

    // The same object through s3:// (anonymous), with the header cache: one open, then none
    let mut config = crate::S3Config::from_url("s3://pgc-opendata-dems/arcticdem/mosaics/v4.1/2m/07_40/07_40_2_2_2m_v4.1_dem.tif").unwrap();
    config.skip_signature = true;
    config.region = Some("us-west-2".into());
    let s3 = crate::ObjectStoreRangeReader::open_s3(config, &IoOptions::default()).await.unwrap();
    let via_s3 = CogReader::from_async_reader_with_hint(Arc::new(s3), Hint::AllUsable).await.unwrap();
    assert_eq!(via_s3.metadata.tile_offsets, m.tile_offsets);
    assert_eq!(via_s3.overviews.len(), 6);
}
