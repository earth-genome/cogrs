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
    assert_eq!(tile.overview_used, None, "{name}: expected full resolution");
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
