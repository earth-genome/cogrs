//! Benchmarks for cogrs tile extraction and point query performance.
//!
//! Run with: `cargo bench`
//!
//! These benchmarks measure the critical hot paths:
//! - XYZ tile extraction at various zoom levels
//! - Point queries (single and batch)
//! - Coordinate transformation
//!
//! Input COG: set `COGRS_BENCH_COG=/path/to/file.tif` to benchmark a real
//! file (the benches assume world coverage in EPSG:3857; the run fails loudly
//! if the path does not exist). When unset, a deterministic synthetic
//! 4096x4096 single-band Deflate GeoTIFF covering the whole Web Mercator
//! world is generated once into a temp dir. That file is striped with no
//! overviews (not a true COG layout), so it benchmarks the full-resolution
//! strip path; set `COGRS_BENCH_COG` to a real tiled COG with overviews for
//! representative COG numbers.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use std::sync::{Arc, LazyLock};
use tokio::runtime::Runtime;

use cogrs::{
    CogReader, BoundingBox, CoordTransformer, PointQuery,
    TileExtractor, ReprojectedRaster, GeoTiffCompression,
    range_reader::LocalRangeReader,
};

/// Name of the env var pointing at a user-supplied COG.
const BENCH_COG_ENV: &str = "COGRS_BENCH_COG";

/// Side length (pixels) of the synthetic world-covering COG.
const SYNTH_SIZE: usize = 4096;

/// Keeps the temp dir alive for the whole process; second field is the COG path.
static TEST_COG: LazyLock<(Option<tempfile::TempDir>, String)> = LazyLock::new(|| {
    if let Some(p) = std::env::var_os(BENCH_COG_ENV) {
        let p = std::path::PathBuf::from(p);
        assert!(
            p.is_file(),
            "{BENCH_COG_ENV} is set to {p:?}, but that file does not exist"
        );
        return (None, p.to_string_lossy().into_owned());
    }
    let dir = tempfile::tempdir().expect("create temp dir for synthetic COG");
    let path = dir.path().join("synthetic_world_3857.tif");
    let bounds = BoundingBox::from_xyz(0, 0, 0);
    let res = (bounds.maxx - bounds.minx) / SYNTH_SIZE as f64;
    let mut pixels = Vec::with_capacity(SYNTH_SIZE * SYNTH_SIZE);
    for y in 0..SYNTH_SIZE {
        for x in 0..SYNTH_SIZE {
            let (fx, fy) = (x as f32 / SYNTH_SIZE as f32, y as f32 / SYNTH_SIZE as f32);
            pixels.push(
                100.0 + 50.0 * (fx * 12.0).sin() * (fy * 9.0).cos() + 30.0 * fx + 20.0 * fy,
            );
        }
    }
    let raster = ReprojectedRaster {
        pixels,
        bands: 1,
        width: SYNTH_SIZE,
        height: SYNTH_SIZE,
        crs: 3857,
        bounds,
        resolution: (res, res),
        nodata: None,
    };
    raster
        .write_geotiff_compressed(&path, GeoTiffCompression::Deflate)
        .expect("write synthetic COG");
    let path = path.to_string_lossy().into_owned();
    (Some(dir), path)
});

/// Path to the COG used by the file-backed benchmarks (see module docs).
fn test_cog_path() -> &'static str {
    &TEST_COG.1
}

/// Benchmark XYZ tile extraction at various zoom levels
fn bench_xyz_tile_extraction(c: &mut Criterion) {
    let path = test_cog_path();

    let rt = Runtime::new().unwrap();
    let reader = LocalRangeReader::new(path).unwrap();
    let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

    let mut group = c.benchmark_group("xyz_tile_extraction");

    // Benchmark at different zoom levels
    for zoom in [0, 2, 4, 6, 8] {
        // Use center tile at each zoom level
        let max_tile = 2u32.pow(zoom);
        let x = max_tile / 2;
        let y = max_tile / 2;

        group.bench_with_input(
            BenchmarkId::new("zoom", zoom),
            &(zoom, x, y),
            |b, &(z, x, y)| {
                b.iter(|| {
                    rt.block_on(TileExtractor::new(black_box(&cog)).xyz(z, x, y).output_size(256, 256).extract())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark tile extraction with different output sizes
fn bench_tile_sizes(c: &mut Criterion) {
    let path = test_cog_path();

    let rt = Runtime::new().unwrap();
    let reader = LocalRangeReader::new(path).unwrap();
    let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

    let mut group = c.benchmark_group("tile_sizes");
    let extent = BoundingBox::from_xyz(4, 8, 8);

    for size in [128, 256, 512, 1024] {
        group.bench_with_input(
            BenchmarkId::new("size", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(TileExtractor::new(black_box(&cog)).bounds(extent).output_size(size, size).extract())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark point queries
fn bench_point_query(c: &mut Criterion) {
    let path = test_cog_path();

    let reader = LocalRangeReader::new(path).unwrap();
    let cog = CogReader::from_reader(Arc::new(reader)).unwrap();

    let mut group = c.benchmark_group("point_query");

    // Single point query
    group.bench_function("single_lonlat", |b| {
        b.iter(|| {
            cog.sample_lonlat(black_box(0.0), black_box(0.0))
        });
    });

    // Batch point queries
    let points: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (-122.4, 37.8),
        (139.7, 35.7),
        (2.3, 48.9),
        (-43.2, -22.9),
        (10.0, 51.5),
        (-74.0, 40.7),
        (116.4, 39.9),
        (37.6, 55.7),
        (151.2, -33.9),
    ];

    group.bench_function("batch_10_points", |b| {
        b.iter(|| {
            cog.sample_points_lonlat(black_box(&points))
        });
    });

    group.finish();
}

/// Benchmark coordinate transformation
fn bench_coord_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("coord_transform");

    // Test creating transformers
    group.bench_function("create_4326_to_3857", |b| {
        b.iter(|| {
            CoordTransformer::new(black_box(4326), black_box(3857))
        });
    });

    // Test transforming coordinates
    let transformer = CoordTransformer::new(4326, 3857).unwrap();

    group.bench_function("transform_single", |b| {
        b.iter(|| {
            transformer.transform(black_box(-122.4), black_box(37.8))
        });
    });

    // Batch transform
    let points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let lon = -180.0 + (i as f64 * 3.6);
            let lat = (i as f64 * 0.9) - 45.0;
            (lon, lat)
        })
        .collect();

    group.bench_function("transform_batch_100", |b| {
        b.iter(|| {
            transformer.transform_batch(black_box(&points))
        });
    });

    group.finish();
}

/// Benchmark BoundingBox operations
fn bench_bounding_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounding_box");

    group.bench_function("from_xyz_z0", |b| {
        b.iter(|| {
            BoundingBox::from_xyz(black_box(0), black_box(0), black_box(0))
        });
    });

    group.bench_function("from_xyz_z10", |b| {
        b.iter(|| {
            BoundingBox::from_xyz(black_box(10), black_box(512), black_box(512))
        });
    });

    group.finish();
}

/// Benchmark COG file opening (metadata parsing)
fn bench_cog_open(c: &mut Criterion) {
    let path = test_cog_path();

    c.bench_function("cog_open", |b| {
        b.iter(|| {
            let reader = LocalRangeReader::new(black_box(&path)).unwrap();
            CogReader::from_reader(Arc::new(reader))
        });
    });
}

criterion_group!(
    benches,
    bench_xyz_tile_extraction,
    bench_tile_sizes,
    bench_point_query,
    bench_coord_transform,
    bench_bounding_box,
    bench_cog_open,
);

criterion_main!(benches);
