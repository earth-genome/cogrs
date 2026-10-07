//! Golden-output regression tests.
//!
//! These pin the *observable results* of the read/extract pipeline (metadata, decoded tiles,
//! extracted XYZ tiles, reprojections, point samples, error text) as digests recorded in
//! `tests/golden/*.txt`. They were captured from the synchronous-I/O implementation and must
//! keep passing unchanged through the async-I/O refactor.
//!
//! * `golden_synthetic` builds its inputs with [`crate::test_support`] and always runs.
//! * `golden_fixtures` uses the (uncommitted) `tests/data/*.tif` fixtures and skips when absent.
//!
//! Regenerate with `COGRS_UPDATE_GOLDEN=1 cargo test golden_` (only when a behaviour change is
//! intended and reviewed).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use crate::test_support::{build_cog, CogSpec, Sample};
use crate::{
    BoundingBox, CogReader, MemoryRangeReader, OverviewQualityHint, PointQuery, Reprojector,
    ResamplingMethod, TileData, TileExtractor,
};

const METHODS: [(&str, ResamplingMethod); 3] = [
    ("nearest", ResamplingMethod::Nearest),
    ("bilinear", ResamplingMethod::Bilinear),
    ("bicubic", ResamplingMethod::Bicubic),
];

/// FNV-1a over the f32 bit patterns, with every NaN canonicalised.
fn digest(values: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        let bits = if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() };
        for b in bits.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

fn digest_u64(values: &[u64]) -> String {
    digest(&values.iter().flat_map(|v| [*v as u32 as f32, (*v >> 32) as u32 as f32]).collect::<Vec<_>>())
}

fn summarize_values(values: &[f32]) -> String {
    let valid = values.iter().filter(|v| !v.is_nan()).count();
    format!("n={} valid={valid} digest={}", values.len(), digest(values))
}

fn summarize_tile(t: &TileData) -> String {
    format!(
        "{}x{}x{} bytes={} tiles={} ovr={:?} nodata={:?} {}",
        t.width,
        t.height,
        t.bands,
        t.bytes_fetched,
        t.tiles_read,
        t.overview_used,
        t.nodata,
        summarize_values(&t.pixels)
    )
}

/// Collects `key -> value` lines and compares them with / writes the golden file.
struct Golden {
    file: &'static str,
    lines: Vec<(String, String)>,
}

impl Golden {
    fn new(file: &'static str) -> Self {
        Self { file, lines: Vec::new() }
    }

    fn rec(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.lines.push((key.into(), value.into()));
    }

    fn path(&self) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(self.file)
    }

    fn finish(self) {
        let path = self.path();
        if std::env::var_os("COGRS_UPDATE_GOLDEN").is_some() {
            let mut out = String::new();
            for (k, v) in &self.lines {
                writeln!(out, "{k}\t{v}").unwrap();
            }
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, out).unwrap();
            return;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing golden file {}: {e} (COGRS_UPDATE_GOLDEN=1 to create)", path.display()));
        let expected: BTreeMap<&str, &str> = text.lines().filter_map(|l| l.split_once('\t')).collect();
        let mut problems = Vec::new();
        for (k, v) in &self.lines {
            match expected.get(k.as_str()) {
                Some(e) if e == v => {}
                Some(e) => problems.push(format!("{k}\n    expected: {e}\n    actual:   {v}")),
                None => problems.push(format!("{k}\n    not in golden file; actual: {v}")),
            }
        }
        let seen: std::collections::HashSet<&str> = self.lines.iter().map(|(k, _)| k.as_str()).collect();
        for k in expected.keys() {
            if !seen.contains(k) {
                problems.push(format!("{k}\n    in golden file but not produced"));
            }
        }
        assert!(problems.is_empty(), "{} golden mismatches in {}:\n{}", problems.len(), self.file, problems.join("\n"));
    }
}

fn tile_for(lon: f64, lat: f64, z: u32) -> (u32, u32) {
    let n = f64::from(1u32 << z);
    let x = ((lon + 180.0) / 360.0 * n).floor();
    let lat_rad = lat.to_radians();
    let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor();
    (x as u32, y as u32)
}

fn open_memory(name: &str, spec: &CogSpec, hint: OverviewQualityHint) -> CogReader {
    let bytes = build_cog(spec);
    let reader = MemoryRangeReader::new(bytes, format!("golden://{name}"));
    CogReader::from_reader_with_hint(Arc::new(reader), hint).unwrap()
}

fn record_metadata(g: &mut Golden, case: &str, r: &CogReader) {
    let m = &r.metadata;
    g.rec(
        format!("{case}/meta"),
        format!(
            "{}x{} tile={}x{} across={} down={} bands={} {:?} {:?} predictor={} crs={:?} nodata={:?} tiled={} \
             scale={:?} tie={:?} offsets={} counts={}",
            m.width,
            m.height,
            m.tile_width,
            m.tile_height,
            m.tiles_across,
            m.tiles_down,
            m.bands,
            m.data_type,
            m.compression,
            m.predictor,
            m.crs_code,
            m.nodata,
            m.is_tiled,
            m.geo_transform.pixel_scale,
            m.geo_transform.tiepoint,
            digest_u64(&m.tile_offsets),
            digest_u64(&m.tile_byte_counts),
        ),
    );
    for (i, o) in r.overviews.iter().enumerate() {
        g.rec(
            format!("{case}/overview{i}"),
            format!(
                "{}x{} tile={}x{} across={} down={} scale={} offsets={} counts={}",
                o.width,
                o.height,
                o.tile_width,
                o.tile_height,
                o.tiles_across,
                o.tiles_down,
                o.scale,
                digest_u64(&o.tile_offsets),
                digest_u64(&o.tile_byte_counts),
            ),
        );
    }
    g.rec(format!("{case}/min_usable"), format!("{:?}", r.min_usable_overview));
    g.rec(format!("{case}/hint"), format!("{:?}", r.compute_overview_quality_hint()));
}

async fn record_extracts(g: &mut Golden, case: &str, r: &CogReader, tiles: &[(u32, u32, u32)], bands_subset: &[usize]) {
    for &(z, x, y) in tiles {
        for (mname, m) in METHODS {
            let t = TileExtractor::new(r).xyz(z, x, y).size(128).resampling(m).extract().await;
            g.rec(
                format!("{case}/xyz/{z}/{x}/{y}/{mname}"),
                match t {
                    Ok(t) => summarize_tile(&t),
                    Err(e) => format!("ERR {e}"),
                },
            );
        }
    }
    // A non-square output and a band subset on the first tile.
    if let Some(&(z, x, y)) = tiles.first() {
        let t = TileExtractor::new(r).xyz(z, x, y).output_size(96, 64).extract().await.unwrap();
        g.rec(format!("{case}/xyz/{z}/{x}/{y}/96x64"), summarize_tile(&t));
        if !bands_subset.is_empty() {
            let t = TileExtractor::new(r)
                .xyz(z, x, y)
                .size(100)
                .resampling(ResamplingMethod::Bilinear)
                .bands(bands_subset)
                .extract()
                .await
                .unwrap();
            g.rec(format!("{case}/xyz/{z}/{x}/{y}/bands{bands_subset:?}"), summarize_tile(&t));
        }
    }
}

/// Straddling the raster edge: the tile at `(z, x, y)` shifted by half a tile.
async fn record_shifted_extracts(g: &mut Golden, case: &str, r: &CogReader, z: u32, x: u32, y: u32) {
    let b = BoundingBox::from_xyz(z, x, y);
    let (dx, dy) = ((b.maxx - b.minx) * 0.5, (b.maxy - b.miny) * 0.5);
    for (mname, m) in METHODS {
        let shifted = BoundingBox::new(b.minx + dx, b.miny + dy, b.maxx + dx, b.maxy + dy);
        let t = TileExtractor::new(r).bounds(shifted).size(128).resampling(m).extract().await.unwrap();
        g.rec(format!("{case}/shifted/{z}/{x}/{y}/{mname}"), summarize_tile(&t));
    }
}

fn record_tile_reads(g: &mut Golden, case: &str, r: &CogReader) {
    let n = r.metadata.tile_offsets.len();
    for idx in [0, n / 2, n - 1] {
        let v = r.read_tile(idx).map(|v| summarize_values(&v)).unwrap_or_else(|e| format!("ERR {e}"));
        g.rec(format!("{case}/read_tile/{idx}"), v);
    }
    for (oi, o) in r.overviews.iter().enumerate() {
        let n = o.tile_offsets.len();
        for idx in [0, n - 1] {
            let v = r.read_overview_tile(oi, idx).map(|v| summarize_values(&v)).unwrap_or_else(|e| format!("ERR {e}"));
            g.rec(format!("{case}/read_overview_tile/{oi}/{idx}"), v);
        }
    }
    let (min, max) = r.estimate_min_max_fast().unwrap();
    g.rec(format!("{case}/minmax_fast"), format!("{min} {max}"));
}

fn record_samples(g: &mut Golden, case: &str, r: &CogReader, lonlat: &[(f64, f64)], crs_pts: &[(i32, f64, f64)]) {
    for &(lon, lat) in lonlat {
        let s = r.sample_lonlat(lon, lat).map(|s| {
            let mut v: Vec<_> = s.values.iter().map(|(b, v)| (*b, v.to_bits())).collect();
            v.sort_unstable();
            format!("{v:?}")
        });
        g.rec(format!("{case}/sample_lonlat/{lon}/{lat}"), s.unwrap_or_else(|e| format!("ERR {e}")));
    }
    for &(crs, x, y) in crs_pts {
        let s = r.sample_crs(crs, x, y).map(|s| {
            let mut v: Vec<_> = s.values.iter().map(|(b, v)| (*b, v.to_bits())).collect();
            v.sort_unstable();
            format!("{v:?}")
        });
        g.rec(format!("{case}/sample_crs/{crs}/{x}/{y}"), s.unwrap_or_else(|e| format!("ERR {e}")));
    }
}

async fn record_reproject(g: &mut Golden, case: &str, r: &CogReader, crs: u32, size: (usize, usize)) {
    let rr = Reprojector::new(r).to_crs(crs).size(size.0, size.1).resampling(ResamplingMethod::Bilinear).extract().await;
    g.rec(
        format!("{case}/reproject/{crs}"),
        match rr {
            Ok(rr) => format!(
                "{}x{}x{} crs={} bounds=({:.6},{:.6},{:.6},{:.6}) nodata={:?} {}",
                rr.width,
                rr.height,
                rr.bands,
                rr.crs,
                rr.bounds.minx,
                rr.bounds.miny,
                rr.bounds.maxx,
                rr.bounds.maxy,
                rr.nodata,
                summarize_values(&rr.pixels)
            ),
            Err(e) => format!("ERR {e}"),
        },
    );
    // Streaming: digest of every chunk in order.
    let streaming = Reprojector::new(r).to_crs(crs).size(size.0, size.1).streaming(48);
    let mut parts = Vec::new();
    streaming
        .for_each_chunk(|pos, chunk| {
            parts.push(format!("{pos:?}:{}x{}:{}", chunk.width, chunk.height, digest(&chunk.pixels)));
            Ok(())
        })
        .await
        .unwrap();
    g.rec(format!("{case}/reproject_streaming/{crs}"), parts.join(" "));
}

fn rgb_u8_spec() -> CogSpec {
    let ext = BoundingBox::from_xyz(3, 2, 3);
    let px = (ext.maxx - ext.minx) / 1000.0;
    CogSpec {
        width: 1000,
        height: 800,
        tile: 64,
        bands: 3,
        sample: Sample::U8,
        deflate: true,
        predictor: true,
        epsg: 3857,
        origin: (ext.minx, ext.maxy),
        pixel_size: (px, px),
        nodata: None,
        overviews: 3,
        sparse: vec![(0, 5), (0, 6), (0, 40), (1, 3)],
        corrupt: vec![],
        pixel: |b, x, y| ((x * 3 + y * 5 + b * 40) % 251) as f64 + 1.0,
    }
}

fn f32_4326_spec() -> CogSpec {
    CogSpec {
        width: 1000,
        height: 800,
        tile: 128,
        bands: 1,
        sample: Sample::F32,
        deflate: true,
        predictor: false,
        epsg: 4326,
        origin: (-10.0, 50.0),
        pixel_size: (0.01, 0.01),
        nodata: Some(-9999.0),
        overviews: 3,
        sparse: vec![(0, 10), (2, 1)],
        corrupt: vec![],
        pixel: |_, x, y| {
            if (x / 100 + y / 100) % 5 == 0 {
                -9999.0
            } else {
                x as f64 * 0.37 + y as f64 * 0.11 + 1.0
            }
        },
    }
}

fn u16_utm_spec() -> CogSpec {
    CogSpec {
        width: 600,
        height: 600,
        tile: 64,
        bands: 1,
        sample: Sample::U16,
        deflate: false,
        predictor: false,
        epsg: 32610,
        origin: (540_000.0, 4_200_000.0),
        pixel_size: (30.0, 30.0),
        nodata: None,
        overviews: 2,
        sparse: vec![],
        corrupt: vec![],
        pixel: |_, x, y| ((x * 13 + y * 7) % 4000) as f64 + 100.0,
    }
}

/// 1030 x 770: the /2, /4 and /8 overviews are 515x385, 258x193 and 129x97, so their size
/// ratios are not integers and differ per axis (7.984 and 7.938 at /8).
fn odd_size_spec() -> CogSpec {
    let ext = BoundingBox::from_xyz(3, 2, 3);
    let px = (ext.maxx - ext.minx) / 1030.0;
    CogSpec {
        width: 1030,
        height: 770,
        tile: 64,
        bands: 1,
        sample: Sample::U16,
        deflate: true,
        predictor: false,
        epsg: 3857,
        origin: (ext.minx, ext.maxy),
        pixel_size: (px, px),
        nodata: None,
        overviews: 3,
        sparse: vec![],
        corrupt: vec![],
        pixel: |_, x, y| ((x * 7 + y * 13) % 4000) as f64 + 1.0,
    }
}

fn corrupt_spec() -> CogSpec {
    let ext = BoundingBox::from_xyz(3, 2, 3);
    let px = (ext.maxx - ext.minx) / 256.0;
    CogSpec {
        width: 256,
        height: 256,
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
        corrupt: vec![(0, 5)],
        pixel: |_, x, y| ((x + y) % 200) as f64 + 1.0,
    }
}

#[tokio::test]
async fn golden_synthetic() {
    let mut g = Golden::new("synthetic.txt");

    // --- 3-band u8, 3857, deflate + predictor, overviews, sparse tiles -----------------------
    let case = "rgb_u8_3857";
    let r = open_memory(case, &rgb_u8_spec(), OverviewQualityHint::ComputeAtRuntime);
    record_metadata(&mut g, case, &r);
    let tiles = [(2, 1, 1), (3, 2, 3), (4, 4, 6), (4, 5, 7), (5, 8, 12), (5, 9, 13), (6, 17, 25), (7, 35, 51)];
    record_extracts(&mut g, case, &r, &tiles, &[2, 0]).await;
    record_shifted_extracts(&mut g, case, &r, 5, 8, 12).await;
    record_tile_reads(&mut g, case, &r);
    let ext = BoundingBox::from_xyz(3, 2, 3);
    let (cx, cy) = ((ext.minx + ext.maxx) / 2.0, (ext.miny + ext.maxy) / 2.0);
    record_samples(&mut g, case, &r, &[], &[(3857, cx, cy), (3857, ext.minx + 10.0, ext.maxy - 10.0), (3857, cx, ext.miny - 1.0)]);
    record_reproject(&mut g, case, &r, 4326, (100, 80)).await;
    // Hints bypass the runtime analysis.
    for (hname, hint) in [
        ("none_usable", OverviewQualityHint::NoneUsable),
        ("all_usable", OverviewQualityHint::AllUsable),
        ("min_usable_1", OverviewQualityHint::MinUsable(1)),
    ] {
        let rh = open_memory(&format!("{case}_{hname}"), &rgb_u8_spec(), hint);
        g.rec(format!("{case}/hinted/{hname}/min_usable"), format!("{:?}", rh.min_usable_overview));
        let t = TileExtractor::new(&rh).xyz(4, 4, 6).size(128).extract().await.unwrap();
        g.rec(format!("{case}/hinted/{hname}/xyz/4/4/6"), summarize_tile(&t));
    }

    // --- f32, 4326 source, nodata, sparse tiles ----------------------------------------------
    let case = "f32_4326_nodata";
    let r = open_memory(case, &f32_4326_spec(), OverviewQualityHint::ComputeAtRuntime);
    record_metadata(&mut g, case, &r);
    let mut tiles = Vec::new();
    for z in [4, 5, 6, 7] {
        let (x, y) = tile_for(-5.0, 46.0, z);
        tiles.push((z, x, y));
    }
    let (x8, y8) = tile_for(-9.9, 49.9, 8);
    tiles.push((8, x8, y8));
    record_extracts(&mut g, case, &r, &tiles, &[]).await;
    record_shifted_extracts(&mut g, case, &r, tiles[2].0, tiles[2].1, tiles[2].2).await;
    record_tile_reads(&mut g, case, &r);
    record_samples(&mut g, case, &r, &[(-5.0, 46.0), (-9.99, 49.99), (-4.5, 44.0), (3.0, 46.0)], &[(4326, -6.5, 47.5)]);
    record_reproject(&mut g, case, &r, 3857, (90, 70)).await;
    // Output in WGS84 (bounds are degrees).
    for (mname, m) in METHODS {
        let t = TileExtractor::new(&r)
            .bounds(BoundingBox::new(-8.0, 44.0, -2.0, 48.0))
            .output_crs(4326)
            .size(96)
            .resampling(m)
            .extract()
            .await
            .unwrap();
        g.rec(format!("{case}/out4326/{mname}"), summarize_tile(&t));
    }

    // --- u16, UTM 10N source (proj4rs path), uncompressed -----------------------------------
    let case = "u16_utm10";
    let r = open_memory(case, &u16_utm_spec(), OverviewQualityHint::ComputeAtRuntime);
    record_metadata(&mut g, case, &r);
    let mut tiles = Vec::new();
    for z in [10, 11, 12, 13] {
        let (x, y) = tile_for(-122.4, 37.8, z);
        tiles.push((z, x, y));
        tiles.push((z, x + 1, y));
    }
    record_extracts(&mut g, case, &r, &tiles, &[]).await;
    record_tile_reads(&mut g, case, &r);
    record_samples(&mut g, case, &r, &[(-122.4, 37.8), (-122.3, 37.75)], &[(32610, 550_000.0, 4_190_000.0)]);
    record_reproject(&mut g, case, &r, 4326, (64, 64)).await;

    // --- overviews with non-integer size ratios -----------------------------------------------
    let case = "u16_odd_size_overviews";
    let r = open_memory(case, &odd_size_spec(), OverviewQualityHint::AllUsable);
    record_metadata(&mut g, case, &r);
    for (i, o) in r.overviews.iter().enumerate() {
        g.rec(format!("{case}/ratio{i}"), format!("{}x{} scale={} x={:.9} y={:.9}", o.width, o.height, o.scale, o.scale_x, o.scale_y));
    }
    let tiles = [(2, 1, 1), (3, 2, 3), (4, 4, 6), (4, 5, 7), (5, 8, 12), (5, 9, 13)];
    record_extracts(&mut g, case, &r, &tiles, &[]).await;
    record_tile_reads(&mut g, case, &r);

    // --- read errors keep their text ----------------------------------------------------------
    let case = "corrupt_tile";
    let r = open_memory(case, &corrupt_spec(), OverviewQualityHint::AllUsable);
    let ok = TileExtractor::new(&r).xyz(5, 8, 12).size(64).extract().await;
    g.rec(format!("{case}/avoiding"), ok.map(|t| summarize_tile(&t)).unwrap_or_else(|e| format!("ERR {e}")));
    let err = TileExtractor::new(&r).xyz(3, 2, 3).size(64).extract().await.unwrap_err().to_string();
    g.rec(format!("{case}/err_prefix"), err.split(':').next().unwrap().to_string());
    g.rec(format!("{case}/read_tile_err"), r.read_tile(5).is_err().to_string());

    g.finish();
}

/// Same checks on the real fixtures; skipped when they have not been fetched.
#[tokio::test]
async fn golden_fixtures() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let dem = dir.join("copernicus_dem_san_francisco.tif");
    let rgb = dir.join("natural_earth_rgb.tif");
    if !dem.exists() || !rgb.exists() {
        eprintln!("Skipping golden_fixtures: fixtures not found (see tests/data/README.md)");
        return;
    }
    let mut g = Golden::new("fixtures.txt");

    let case = "copernicus_dem";
    let r = open_fixture(&dem, "golden://copernicus_dem");
    record_metadata(&mut g, case, &r);
    let mut tiles = Vec::new();
    for z in [6, 8, 9, 10, 11, 12] {
        let (x, y) = tile_for(-122.4, 37.8, z);
        tiles.push((z, x, y));
    }
    let (x, y) = tile_for(-122.4, 37.8, 12);
    tiles.push((12, x + 1, y + 1));
    record_extracts(&mut g, case, &r, &tiles, &[]).await;
    record_shifted_extracts(&mut g, case, &r, 10, tiles[3].1, tiles[3].2).await;
    record_tile_reads(&mut g, case, &r);
    record_samples(&mut g, case, &r, &[(-122.4, 37.8), (-122.95, 37.05), (-122.0, 38.0)], &[(4326, -122.5, 37.5)]);
    record_reproject(&mut g, case, &r, 3857, (120, 120)).await;

    let case = "natural_earth_rgb";
    let r = open_fixture(&rgb, "golden://natural_earth_rgb");
    record_metadata(&mut g, case, &r);
    let mut tiles = vec![(0, 0, 0)];
    for z in 1..=3u32 {
        for y in 0..(1u32 << z).min(2) {
            for x in 0..(1u32 << z).min(3) {
                tiles.push((z, x, y + (1 << z) / 2 - 1));
            }
        }
    }
    record_extracts(&mut g, case, &r, &tiles, &[2, 1]).await;
    record_tile_reads(&mut g, case, &r);
    record_samples(&mut g, case, &r, &[(0.0, 0.0), (-122.4, 37.8), (151.2, -33.9)], &[]);
    record_reproject(&mut g, case, &r, 4326, (144, 72)).await;

    g.finish();
}

/// Open a fixture through a private in-memory copy. Other tests read the same files by path
/// and populate the process-wide tile cache under that path; a unique identifier keeps the
/// goldens (which include bytes fetched) independent of test order.
fn open_fixture(path: &std::path::Path, id: &str) -> CogReader {
    let bytes = std::fs::read(path).unwrap();
    CogReader::from_reader(Arc::new(MemoryRangeReader::new(bytes, id.to_string()))).unwrap()
}
