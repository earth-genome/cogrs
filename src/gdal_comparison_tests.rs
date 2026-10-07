//! Output-pixel convention checks against GDAL (`gdalwarp`) on the real fixtures.
//!
//! For each output pixel, its centre is `min + (i + 0.5) * res`; nearest resampling takes the
//! source pixel *containing* that point and bilinear interpolates between pixel centres, exactly
//! what `gdalwarp -r near` / `-r bilinear` do. Skips when `gdalwarp` or a fixture is missing.
//!
//! Each test prints a `GDALCMP` line with the share of bit-identical pixels and the mean absolute
//! difference (pixels valid in both).

use std::path::Path;
use std::process::Command;

use crate::{BoundingBox, CogReader, ResamplingMethod, TileExtractor};

const NODATA: f32 = -9999.0;

/// Warp `src` into `bounds` (EPSG:3857, `size` x `size`) with `gdalwarp -r <alg>` as Float32.
fn gdalwarp(src: &str, bounds: &BoundingBox, size: usize, alg: &str) -> Option<Vec<f32>> {
    let dir = tempfile::tempdir().ok()?;
    let out = dir.path().join("ref.tif");
    let status = Command::new("gdalwarp")
        .args(["-q", "-overwrite", "-t_srs", "EPSG:3857", "-te"])
        .args([bounds.minx, bounds.miny, bounds.maxx, bounds.maxy].map(|v| format!("{v:.9}")))
        .args(["-ts", &size.to_string(), &size.to_string(), "-r", alg, "-et", "0"])
        .args(["-ot", "Float32", "-dstnodata", "-9999", "-co", "COMPRESS=NONE"])
        .arg(src)
        .arg(&out)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let ds = gdal::Dataset::open(&out).ok()?;
    let buf: gdal::raster::Buffer<f32> = ds.rasterband(1).ok()?.read_as((0, 0), (size, size), (size, size), None).ok()?;
    Some(buf.data().to_vec())
}

struct Comparison {
    identical_pct: f64,
    mean_abs_diff: f64,
    max_abs_diff: f64,
    compared: usize,
    validity_mismatch: usize,
}

fn compare(ours: &[f32], gdal: &[f32]) -> Comparison {
    let (mut same, mut compared, mut sum, mut max, mut mismatch) = (0usize, 0usize, 0.0f64, 0.0f64, 0usize);
    for (&o, &g) in ours.iter().zip(gdal) {
        let (o_valid, g_valid) = (!o.is_nan(), g != NODATA);
        if o_valid != g_valid {
            mismatch += 1;
            continue;
        }
        if o_valid {
            compared += 1;
            let d = f64::from((o - g).abs());
            same += usize::from(o.to_bits() == g.to_bits());
            sum += d;
            max = max.max(d);
        }
    }
    Comparison {
        identical_pct: 100.0 * same as f64 / compared.max(1) as f64,
        mean_abs_diff: sum / compared.max(1) as f64,
        max_abs_diff: max,
        compared,
        validity_mismatch: mismatch,
    }
}

async fn run(name: &str, path: &str, bounds: &BoundingBox, method: ResamplingMethod, alg: &str) -> Option<Comparison> {
    run_at(name, path, bounds, method, alg, None).await
}

/// [`run`] for a window that both cogrs and GDAL serve from overview `expected_overview`.
async fn run_at(
    name: &str,
    path: &str,
    bounds: &BoundingBox,
    method: ResamplingMethod,
    alg: &str,
    expected_overview: Option<usize>,
) -> Option<Comparison> {
    // Compares band 0 (gdalwarp's output band 1).
    if !Path::new(path).exists() {
        println!("Skipping {name}: {path} not found");
        return None;
    }
    let Some(reference) = gdalwarp(path, bounds, 256, alg) else {
        println!("Skipping {name}: gdalwarp not available");
        return None;
    };
    let reader = CogReader::open(path).expect("open");
    let tile =
        TileExtractor::new(&reader).bounds(*bounds).size(256).resampling(method).bands(&[0]).extract().await.expect("extract");
    assert_eq!(tile.overview_used, expected_overview, "{name}: unexpected source level");
    let c = compare(&tile.pixels, &reference);
    println!(
        "GDALCMP {name}: {:.2}% bit-identical, mean|diff| {:.4}, max|diff| {:.3}, {} compared, {} validity mismatches",
        c.identical_pct, c.mean_abs_diff, c.max_abs_diff, c.compared, c.validity_mismatch
    );
    Some(c)
}

/// `PixelIsArea` fixture (gray_3857, 1911 m pixels), same CRS, upsampled about 2x.
fn gray_window() -> BoundingBox {
    let (minx, maxy, res) = (-3_000_000.37, 4_000_000.81, 900.3);
    BoundingBox::new(minx, maxy - 256.0 * res, minx + 256.0 * res, maxy)
}

/// `PixelIsPoint` fixture (Copernicus DEM, 4326, ~30 m pixels) warped to 3857, upsampled about 2.5x.
fn dem_window() -> BoundingBox {
    let (cx, cy) = crate::lon_lat_to_mercator(-122.4, 37.8);
    let res = 12.3;
    BoundingBox::new(cx + 0.37, cy + 0.81 - 256.0 * res, cx + 0.37 + 256.0 * res, cy + 0.81)
}

/// `PixelIsArea` RGB fixture with real texture (Natural Earth, 4326, 0.25 degree pixels), warped to
/// 3857 and upsampled about 2.3x.
fn world_window() -> BoundingBox {
    let (cx, cy) = crate::lon_lat_to_mercator(10.0, 45.0);
    let res = 12_000.0;
    BoundingBox::new(cx + 0.37, cy + 0.81 - 256.0 * res, cx + 0.37 + 256.0 * res, cy + 0.81)
}

const WORLD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/natural_earth_rgb.tif");
const GRAY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/grayscale/gray_3857-cog.tif");
const DEM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/copernicus_dem_san_francisco.tif");

#[tokio::test]
async fn nearest_matches_gdalwarp_on_a_pixel_is_area_raster() {
    let Some(c) = run("gray_3857 near", GRAY, &gray_window(), ResamplingMethod::Nearest, "near").await else { return };
    assert!(c.compared > 50_000);
    assert!(c.identical_pct > 99.9, "{:.2}% identical", c.identical_pct);
    assert_eq!(c.validity_mismatch, 0);
}

#[tokio::test]
async fn nearest_matches_gdalwarp_on_a_pixel_is_point_raster() {
    let Some(c) = run("copernicus_dem near", DEM, &dem_window(), ResamplingMethod::Nearest, "near").await else { return };
    assert!(c.compared > 50_000);
    assert!(c.identical_pct > 99.0, "{:.2}% identical", c.identical_pct);
}

#[tokio::test]
async fn bilinear_matches_gdalwarp_on_a_pixel_is_area_raster() {
    let Some(c) = run("gray_3857 bilinear", GRAY, &gray_window(), ResamplingMethod::Bilinear, "bilinear").await else { return };
    assert!(c.mean_abs_diff < 0.05, "mean |diff| {:.4}", c.mean_abs_diff);
}

#[tokio::test]
async fn bilinear_matches_gdalwarp_on_a_pixel_is_point_raster() {
    let Some(c) = run("copernicus_dem bilinear", DEM, &dem_window(), ResamplingMethod::Bilinear, "bilinear").await else { return };
    assert!(c.mean_abs_diff < 0.05, "mean |diff| {:.4}", c.mean_abs_diff);
}

#[tokio::test]
async fn nearest_matches_gdalwarp_on_a_textured_pixel_is_area_raster() {
    let Some(c) = run("natural_earth near", WORLD, &world_window(), ResamplingMethod::Nearest, "near").await else { return };
    assert!(c.compared > 50_000);
    assert!(c.identical_pct > 99.0, "{:.2}% identical", c.identical_pct);
}

#[tokio::test]
async fn bilinear_matches_gdalwarp_on_a_textured_pixel_is_area_raster() {
    let Some(c) = run("natural_earth bilinear", WORLD, &world_window(), ResamplingMethod::Bilinear, "bilinear").await else { return };
    assert!(c.mean_abs_diff < 0.5, "mean |diff| {:.4}", c.mean_abs_diff);
}

/// Synthetic UTM 10N raster (30 m pixels, `size` pixels square) written to `dir` for `gdalwarp`.
fn utm10_raster(dir: &Path, size: usize) -> String {
    use crate::test_support::{build_cog, CogSpec, Sample};
    let spec = CogSpec {
        width: size,
        height: size,
        tile: 64,
        bands: 1,
        sample: Sample::U16,
        deflate: true,
        predictor: false,
        epsg: 32610,
        origin: (545_000.0, 4_195_000.0),
        pixel_size: (30.0, 30.0),
        nodata: None,
        overviews: 0,
        sparse: vec![],
        corrupt: vec![],
        pixel: |_, x, y| ((x * 31 + y * 17) % 4000) as f64 + 1.0,
    };
    let path = dir.join("utm10.tif");
    std::fs::write(&path, build_cog(&spec)).unwrap();
    path.to_str().unwrap().to_string()
}

/// 256 px at 60 m around the centre of a `size` pixel raster; across that width the grid rotation
/// moves a pixel's source row by about 8 source pixels.
fn utm10_window(size: usize) -> BoundingBox {
    let half = size as f64 * 15.0;
    let (cx, cy) = crate::geometry::projection::project_point(32610, 3857, 545_000.0 + half, 4_195_000.0 - half).unwrap();
    let res = 60.0;
    BoundingBox::new(cx - 128.37 * res, cy - 128.81 * res, cx + 127.63 * res, cy + 127.19 * res)
}

/// A UTM raster reprojected to 3857: UTM grid north is rotated against the web mercator grid by
/// the meridian convergence (about 4.6 source pixels over a z10 tile here), so a pixel's source
/// row depends on its column. Synthetic raster written to disk for `gdalwarp`.
#[tokio::test]
async fn nearest_matches_gdalwarp_through_a_rotated_grid() {
    // 12 km square: the window covers the raster and a margin
    let dir = tempfile::tempdir().unwrap();
    let path = utm10_raster(dir.path(), 400);
    let Some(c) = run("synthetic utm10 near", &path, &utm10_window(400), ResamplingMethod::Nearest, "near").await else {
        return;
    };
    assert!(c.compared > 30_000, "only {} pixels compared", c.compared);
    assert!(c.identical_pct > 99.9, "{:.2}% identical", c.identical_pct);
    assert_eq!(c.validity_mismatch, 0);
}

/// A rotated grid downsampled about 1.6 x 1.6 (60 m mercator pixels are 47 m on the ground). The
/// raster (24 km square) covers the whole window: GDAL divides the output size by the part of the
/// source window inside the raster, so a partly covered tile gets a different scale.
#[tokio::test]
async fn bilinear_downsampling_matches_gdalwarp_through_a_rotated_grid() {
    let dir = tempfile::tempdir().unwrap();
    let path = utm10_raster(dir.path(), 800);
    let Some(c) = run("synthetic utm10 bilinear", &path, &utm10_window(800), ResamplingMethod::Bilinear, "bilinear").await else {
        return;
    };
    // Measured: 99.95% bit-identical, max |diff| 0 on values up to 4000
    assert_eq!(c.validity_mismatch, 0);
    assert!(c.identical_pct > 99.0, "{:.2}% identical", c.identical_pct);
    assert!(c.max_abs_diff < 0.01, "max |diff| {}", c.max_abs_diff);
}

// ---------------------------------------------------------------------------------------------
// Downsampling: output pixels covering several source pixels
// ---------------------------------------------------------------------------------------------

/// A same-CRS (3857) float raster with texture at every scale: two smooth waves and per-pixel
/// noise, so a kernel of the wrong width gives visibly different values. Written to `dir`.
fn textured_3857(dir: &Path) -> String {
    use crate::test_support::{build_cog, CogSpec, Sample};
    fn noise(x: usize, y: usize) -> f64 {
        let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
        (h % 10_000) as f64 / 10_000.0
    }
    let spec = CogSpec {
        width: 2048,
        height: 2048,
        tile: 256,
        bands: 1,
        sample: Sample::F32,
        deflate: true,
        predictor: false,
        epsg: 3857,
        origin: (1_000_000.0, 5_000_000.0),
        pixel_size: (10.0, 10.0),
        nodata: None,
        overviews: 0,
        sparse: vec![],
        corrupt: vec![],
        pixel: |_, x, y| 400.0 + 200.0 * (x as f64 * 0.05).sin() * (y as f64 * 0.037).cos() + 150.0 * noise(x, y),
    };
    let path = dir.join("textured_3857.tif");
    std::fs::write(&path, build_cog(&spec)).unwrap();
    path.to_str().unwrap().to_string()
}

/// A 256 px window of the synthetic raster where one output pixel covers `ratio` source pixels,
/// at a sub-pixel offset.
fn textured_window(ratio: f64) -> BoundingBox {
    let res = 10.0 * ratio;
    let (minx, maxy) = (1_000_000.0 + 3_000.37, 5_000_000.0 - 3_000.81);
    BoundingBox::new(minx, maxy - 256.0 * res, minx + 256.0 * res, maxy)
}

/// Compares each ratio's window of the synthetic raster with `gdalwarp -r <alg>`.
async fn downsample_synthetic(method: ResamplingMethod, alg: &str, ratios: &[f64]) -> Vec<(f64, Comparison)> {
    let dir = tempfile::tempdir().unwrap();
    let path = textured_3857(dir.path());
    let mut out = Vec::new();
    for &ratio in ratios {
        let name = format!("synthetic 3857 {alg} x{ratio}");
        if let Some(c) = run(&name, &path, &textured_window(ratio), method, alg).await {
            assert_eq!(c.compared, 256 * 256, "{name}");
            assert_eq!(c.validity_mismatch, 0, "{name}");
            out.push((ratio, c));
        }
    }
    out
}

/// Same CRS, no resampling of the grid: GDAL's scale is exactly the ratio here, so the scaled
/// bilinear kernel reproduces its values (measured: 100% bit-identical at every ratio). The
/// fixed 2x2 footprint differs by 10 to 18 on average on this texture.
#[tokio::test]
async fn bilinear_downsampling_matches_gdalwarp_on_a_synthetic_raster() {
    for (ratio, c) in downsample_synthetic(ResamplingMethod::Bilinear, "bilinear", &[1.5, 2.5, 4.0]).await {
        assert!(c.identical_pct > 99.0, "x{ratio}: {:.2}% identical", c.identical_pct);
        assert!(c.max_abs_diff < 0.01, "x{ratio}: max |diff| {}", c.max_abs_diff);
    }
}

/// GDAL's `cubic` is Catmull-Rom, cogrs's bicubic is Mitchell (B = C = 1/3), so values differ
/// even where the footprint matches: on this noisy texture by 1.4 to 3.7 on average (range about
/// 250 to 750, noise amplitude 150), max 16. The bound is set between that and the fixed 4x4
/// footprint's 9.5 to 19 mean, 51 to 85 max, so it fails if the kernel is not scaled.
#[tokio::test]
async fn bicubic_downsampling_matches_gdalwarp_on_a_synthetic_raster() {
    for (ratio, c) in downsample_synthetic(ResamplingMethod::Bicubic, "cubic", &[1.5, 2.5, 4.0]).await {
        assert!(c.mean_abs_diff < 5.0, "x{ratio}: mean |diff| {:.3}", c.mean_abs_diff);
        assert!(c.max_abs_diff < 25.0, "x{ratio}: max |diff| {:.3}", c.max_abs_diff);
    }
}

/// Copernicus DEM (4326, `PixelIsPoint`, 3600 px square, overviews from 1800): a window warped to
/// 3857 where one output pixel covers about 1.6 x 1.3 source pixels (x, y), too few for either
/// side to pick an overview. GDAL estimates its scale from the warp window, which is rounded to
/// whole pixels, so the scale can be off by a fraction of a percent.
fn dem_downsampled_window() -> BoundingBox {
    let (cx, cy) = crate::lon_lat_to_mercator(-122.4, 37.8);
    let res = 50.0;
    BoundingBox::new(cx + 0.37, cy + 0.81 - 256.0 * res, cx + 0.37 + 256.0 * res, cy + 0.81)
}

/// Measured: 99.96% bit-identical, max |diff| 0 (metres); the fixed footprint is 72% identical
/// with mean 0.07 and max 5.3 on this terrain.
#[tokio::test]
async fn bilinear_downsampling_matches_gdalwarp_on_the_dem() {
    let Some(c) = run("copernicus_dem bilinear downsampled", DEM, &dem_downsampled_window(), ResamplingMethod::Bilinear, "bilinear").await else { return };
    assert_eq!(c.validity_mismatch, 0);
    assert!(c.identical_pct > 99.0, "{:.2}% identical", c.identical_pct);
    assert!(c.max_abs_diff < 0.01, "max |diff| {}", c.max_abs_diff);
}

/// Natural Earth RGB (4326, 0.25 degree pixels) warped to 3857 around 45 N, where one output pixel
/// covers about 1.6 x 1.1 source pixels (x, y).
fn world_downsampled_window() -> BoundingBox {
    let (cx, cy) = crate::lon_lat_to_mercator(10.0, 45.0);
    let res = 45_000.0;
    BoundingBox::new(cx + 0.37, cy + 0.81 - 256.0 * res, cx + 0.37 + 256.0 * res, cy + 0.81)
}

/// Measured (8-bit data compared as floats): mean |diff| 0.002, max 0.06; the fixed footprint
/// has mean 1.15 and max 40. The residue is GDAL's whole-pixel source window (scale estimate).
#[tokio::test]
async fn bilinear_downsampling_matches_gdalwarp_on_a_textured_raster() {
    let Some(c) = run("natural_earth bilinear downsampled", WORLD, &world_downsampled_window(), ResamplingMethod::Bilinear, "bilinear").await else { return };
    assert_eq!(c.validity_mismatch, 0);
    assert!(c.mean_abs_diff < 0.01, "mean |diff| {:.4}", c.mean_abs_diff);
    assert!(c.max_abs_diff < 0.5, "max |diff| {:.3}", c.max_abs_diff);
}

/// Measured: mean |diff| 0.56, max 11 (Catmull-Rom against Mitchell, see the synthetic test);
/// the fixed footprint has mean 0.93 and max 26.
#[tokio::test]
async fn bicubic_downsampling_matches_gdalwarp_on_a_textured_raster() {
    let Some(c) = run("natural_earth cubic downsampled", WORLD, &world_downsampled_window(), ResamplingMethod::Bicubic, "cubic").await else { return };
    assert_eq!(c.validity_mismatch, 0);
    assert!(c.mean_abs_diff < 0.75, "mean |diff| {:.4}", c.mean_abs_diff);
    assert!(c.max_abs_diff < 15.0, "max |diff| {:.3}", c.max_abs_diff);
}
