//! Lossless WebP encoding of extracted tiles.
//!
//! Adds [`TileData::to_webp`] / [`TileData::to_webp_with`], turning the output
//! of [`TileExtractor`](crate::TileExtractor) straight into RGBA8 WebP bytes
//! using the pure-Rust `image` crate encoder (no libwebp, no C dependencies).
//!
//! # Output format
//!
//! The result is always an RGBA8 lossless WebP, regardless of source layout:
//!
//! | Source bands | Mapping                                              |
//! |--------------|------------------------------------------------------|
//! | 1            | gray: `R = G = B = band0`, opaque                    |
//! | 3            | `R, G, B` opaque                                     |
//! | 4            | `R, G, B` plus the source alpha band                 |
//! | other        | error (select 1, 3 or 4 bands with `.bands(&[..])`)  |
//!
//! # Transparency
//!
//! A pixel becomes fully transparent (`alpha = 0`) when
//!
//! - any of its bands is `NaN`, or
//! - **all** of its bands equal the nodata value.
//!
//! This is the same all-bands rule the extractor applies to the centre of its downsampling
//! kernel (an input pixel is nodata only if every selected band is), as `gdalwarp` does.
//!
//! The nodata value is taken from [`TileData::nodata`] (copied from the COG
//! metadata at extraction time) unless [`WebpOptions::nodata`] overrides it.
//! For 4-band sources the resulting alpha is `min(source alpha, mask)`.
//!
//! Edge tiles: the extractor fills output pixels outside the COG extent (and in
//! sparse source tiles) with the COG's nodata value, or `NaN` if the COG
//! declares none, and bilinear/bicubic resampling never blends invalid samples
//! into valid ones. Both therefore come out transparent without any caller
//! action. The one exception is [`WebpOptions::nodata`] set to a value
//! *different* from the COG's own nodata: the extractor's fill (the COG's
//! nodata, if declared) then no longer matches and renders as opaque colour.
//!
//! # Value mapping
//!
//! Sample values are mapped `f32 -> u8` by rounding and clamping to `0..=255`,
//! which is exact for `uint8` sources. For 16-bit or floating point sources
//! pass [`WebpOptions::rescale`] to map an explicit `(min, max)` range
//! linearly onto `0..=255`. There is no automatic stretching. The rescale
//! applies to colour/gray bands only; a source alpha band is always
//! round-and-clamped.

use image::codecs::webp::WebPEncoder;
use image::ExtendedColorType;

use crate::tiff_utils::AnyResult;
use crate::xyz_tile::TileData;

/// Options for [`TileData::to_webp_with`].
///
/// The default (`WebpOptions::default()`) uses the tile's own nodata value and
/// plain round-and-clamp value mapping, which is what [`TileData::to_webp`] does.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WebpOptions {
    /// Override the nodata value used for transparency masking.
    ///
    /// `None` uses [`TileData::nodata`]. Pixels containing `NaN` are always
    /// transparent regardless of this setting.
    pub nodata: Option<f64>,
    /// Linear `(min, max)` rescale mapped onto `0..=255` (`min -> 0`,
    /// `max -> 255`, values outside are clamped). `None` rounds and clamps raw
    /// values. `max` must be greater than `min` and both must be finite.
    pub rescale: Option<(f32, f32)>,
}

impl TileData {
    /// Encode this tile as a lossless RGBA8 WebP using the tile's own nodata
    /// value and round-and-clamp value mapping.
    ///
    /// See the [module documentation](crate::webp) for band layouts and
    /// transparency rules.
    ///
    /// # Errors
    /// Returns an error if the band count is not 1, 3 or 4, the dimensions are
    /// zero or inconsistent with `pixels`, or encoding fails.
    pub fn to_webp(&self) -> AnyResult<Vec<u8>> {
        self.to_webp_with(&WebpOptions::default())
    }

    /// Encode this tile as a lossless RGBA8 WebP with explicit [`WebpOptions`].
    ///
    /// # Errors
    /// Returns an error if the band count is not 1, 3 or 4, the dimensions are
    /// zero or inconsistent with `pixels`, the rescale range is invalid, or
    /// encoding fails.
    pub fn to_webp_with(&self, options: &WebpOptions) -> AnyResult<Vec<u8>> {
        let rgba = self.to_rgba8(options)?;
        let width = u32::try_from(self.width).map_err(|_| "Tile width exceeds u32")?;
        let height = u32::try_from(self.height).map_err(|_| "Tile height exceeds u32")?;

        let mut out = Vec::with_capacity(rgba.len() / 2);
        WebPEncoder::new_lossless(&mut out).encode(&rgba, width, height, ExtendedColorType::Rgba8)?;
        Ok(out)
    }

    /// Convert to a single interleaved RGBA8 buffer (`width * height * 4` bytes).
    fn to_rgba8(&self, options: &WebpOptions) -> AnyResult<Vec<u8>> {
        if !matches!(self.bands, 1 | 3 | 4) {
            return Err(format!(
                "WebP encoding supports 1, 3 or 4 bands, got {} (use TileExtractor::bands to select)",
                self.bands
            )
            .into());
        }
        let pixel_count = self
            .width
            .checked_mul(self.height)
            .filter(|&n| n > 0)
            .ok_or("Tile has zero or overflowing dimensions")?;
        let expected = pixel_count
            .checked_mul(self.bands)
            .ok_or("Tile has zero or overflowing dimensions")?;
        if self.pixels.len() != expected {
            return Err(format!(
                "Tile pixel buffer has {} values, expected {} ({}x{}x{})",
                self.pixels.len(),
                expected,
                self.width,
                self.height,
                self.bands
            )
            .into());
        }

        // Value mapping: (offset, scale) applied as `(v - offset) * scale`.
        let (offset, scale) = match options.rescale {
            Some((min, max)) => {
                if !(min.is_finite() && max.is_finite() && max > min) {
                    return Err(format!("Invalid rescale range ({min}, {max}): need finite min < max").into());
                }
                (min, 255.0 / (max - min))
            }
            None => (0.0, 1.0),
        };
        #[allow(clippy::cast_possible_truncation)]
        let nodata = options.nodata.or(self.nodata).map(|v| v as f32);

        // NaN is handled by the caller (masked), so the cast never sees NaN.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let to_u8 = |v: f32| ((v - offset) * scale).round().clamp(0.0, 255.0) as u8;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let alpha_u8 = |v: f32| v.round().clamp(0.0, 255.0) as u8;

        let mut rgba = Vec::with_capacity(pixel_count * 4);
        for px in self.pixels.chunks_exact(self.bands) {
            let masked = px.iter().any(|v| v.is_nan()) || nodata.is_some_and(|nd| px.iter().all(|&v| v == nd));
            if masked {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            match *px {
                [g] => {
                    let g = to_u8(g);
                    rgba.extend_from_slice(&[g, g, g, 255]);
                }
                [r, g, b] => rgba.extend_from_slice(&[to_u8(r), to_u8(g), to_u8(b), 255]),
                [r, g, b, a] => rgba.extend_from_slice(&[to_u8(r), to_u8(g), to_u8(b), alpha_u8(a)]),
                _ => unreachable!("band count validated above"),
            }
        }
        Ok(rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(pixels: Vec<f32>, bands: usize, width: usize, height: usize, nodata: Option<f64>) -> TileData {
        TileData {
            pixels,
            bands,
            width,
            height,
            bytes_fetched: 0,
            tiles_read: 0,
            overview_used: None,
            nodata,
        }
    }

    fn decode(bytes: &[u8]) -> image::RgbaImage {
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WEBP");
        image::load_from_memory(bytes).unwrap().to_rgba8()
    }

    #[test]
    fn nodata_pixel_transparent_data_pixel_opaque() {
        // 2x1 RGB: first pixel all == nodata (0), second has data.
        let t = tile(vec![0.0, 0.0, 0.0, 10.0, 20.0, 30.0], 3, 2, 1, Some(0.0));
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0[3], 0);
        assert_eq!(img.get_pixel(1, 0).0, [10, 20, 30, 255]);
    }

    #[test]
    fn partial_nodata_pixel_stays_opaque() {
        // Only one band equals nodata: not a nodata pixel.
        let t = tile(vec![0.0, 5.0, 0.0], 3, 1, 1, Some(0.0));
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [0, 5, 0, 255]);
    }

    #[test]
    fn no_nodata_keeps_zero_pixels_opaque() {
        let t = tile(vec![0.0, 0.0, 0.0], 3, 1, 1, None);
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 255]);
    }

    #[test]
    fn nodata_override_replaces_tile_nodata() {
        let t = tile(vec![0.0, 0.0, 0.0, 255.0, 255.0, 255.0], 3, 2, 1, Some(0.0));
        let opts = WebpOptions { nodata: Some(255.0), ..Default::default() };
        let img = decode(&t.to_webp_with(&opts).unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 255]);
        assert_eq!(img.get_pixel(1, 0).0[3], 0);
    }

    #[test]
    fn nan_pixel_transparent() {
        let t = tile(vec![f32::NAN, 1.0, 2.0, 7.0, 8.0, 9.0], 3, 2, 1, None);
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0[3], 0);
        assert_eq!(img.get_pixel(1, 0).0, [7, 8, 9, 255]);
    }

    #[test]
    fn negative_nodata_matches() {
        let t = tile(vec![-9999.0, 3.0], 1, 2, 1, Some(-9999.0));
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0[3], 0);
        assert_eq!(img.get_pixel(1, 0).0, [3, 3, 3, 255]);
    }

    #[test]
    fn gray_expands_to_rgb() {
        let t = tile(vec![0.0, 128.0, 255.0, 77.0], 1, 2, 2, None);
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [128, 128, 128, 255]);
        assert_eq!(img.get_pixel(0, 1).0, [255, 255, 255, 255]);
        assert_eq!(img.get_pixel(1, 1).0, [77, 77, 77, 255]);
    }

    #[test]
    fn rgba_source_alpha_preserved_and_combined_with_nodata() {
        let t = tile(
            vec![
                1.0, 2.0, 3.0, 128.0, // semi-transparent source
                4.0, 5.0, 6.0, 0.0, // transparent source
                9.0, 9.0, 9.0, 9.0, // all bands == nodata
            ],
            4,
            3,
            1,
            Some(9.0),
        );
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [1, 2, 3, 128]);
        assert_eq!(img.get_pixel(1, 0).0[3], 0);
        assert_eq!(img.get_pixel(2, 0).0[3], 0);
    }

    #[test]
    fn out_of_range_values_clamped_and_rounded() {
        let t = tile(vec![-5.0, 300.0, 1.4, 1.6, 254.6, 1e9], 3, 2, 1, None);
        let img = decode(&t.to_webp().unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [0, 255, 1, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [2, 255, 255, 255]);
    }

    #[test]
    fn invalid_band_counts_error() {
        for bands in [0usize, 2, 5] {
            let t = tile(vec![0.0; 4 * bands.max(1)], bands, 2, 2, None);
            let err = t.to_webp().unwrap_err().to_string();
            assert!(err.contains("1, 3 or 4 bands"), "bands={bands}: {err}");
        }
    }

    #[test]
    fn inconsistent_buffer_or_dimensions_error() {
        assert!(tile(vec![0.0; 5], 1, 2, 2, None).to_webp().is_err());
        assert!(tile(vec![], 1, 0, 0, None).to_webp().is_err());
    }

    #[test]
    fn rescale_maps_boundaries() {
        // Range 1000..3000 -> 0..255
        let t = tile(vec![1000.0, 1400.0, 3000.0, 500.0, 4000.0, 1000.0 + 2000.0 / 255.0], 3, 2, 1, None);
        let opts = WebpOptions { rescale: Some((1000.0, 3000.0)), ..Default::default() };
        let img = decode(&t.to_webp_with(&opts).unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [0, 51, 255, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [0, 255, 1, 255]); // below min, above max, one step above min
    }

    #[test]
    fn rescale_does_not_touch_alpha() {
        let t = tile(vec![100.0, 100.0, 100.0, 200.0], 4, 1, 1, None);
        let opts = WebpOptions { rescale: Some((0.0, 1000.0)), ..Default::default() };
        let img = decode(&t.to_webp_with(&opts).unwrap());
        assert_eq!(img.get_pixel(0, 0).0, [26, 26, 26, 200]);
    }

    #[test]
    fn invalid_rescale_errors() {
        let t = tile(vec![1.0], 1, 1, 1, None);
        for range in [(5.0, 5.0), (6.0, 5.0), (f32::NAN, 1.0), (0.0, f32::INFINITY)] {
            let opts = WebpOptions { rescale: Some(range), ..Default::default() };
            assert!(t.to_webp_with(&opts).is_err(), "{range:?}");
        }
    }

    #[test]
    fn full_tile_roundtrip_is_bit_exact() {
        let (w, h) = (256usize, 256usize);
        let mut pixels = Vec::with_capacity(w * h * 3);
        for i in 0..w * h {
            pixels.extend_from_slice(&[(i % 251) as f32 + 1.0, ((i / 7) % 255) as f32 + 1.0, ((i * 3) % 253) as f32 + 1.0]);
        }
        let t = tile(pixels.clone(), 3, w, h, Some(0.0));
        let img = decode(&t.to_webp().unwrap());
        for (i, p) in img.pixels().enumerate() {
            let s = &pixels[i * 3..i * 3 + 3];
            assert_eq!(p.0, [s[0] as u8, s[1] as u8, s[2] as u8, 255]);
        }
    }
}
