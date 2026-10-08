//! Undoing the TIFF predictors on decompressed tile bytes, in place: horizontal differencing
//! (predictor 2, TIFF 6.0 section 14) and floating-point differencing (predictor 3, Adobe
//! Photoshop TIFF Technical Note 3).
//!
//! The single-band byte running sum and the 4-byte floating-point un-shuffle use SSE2 on x86_64
//! (part of its baseline, so no runtime detection); every other case, and every other
//! architecture, uses the scalar loops.

use crate::tiff_utils::AnyResult;

/// Undo `predictor` in place on rows of `tile_width * bands` samples of `bytes_per_sample`
/// bytes each. Afterwards the samples are in the file's byte order (`little_endian`), as if the
/// file had no predictor.
///
/// # Errors
/// An unknown predictor, or a sample size the predictor does not define.
pub(crate) fn undo_predictor(
    data: &mut [u8],
    predictor: u16,
    tile_width: usize,
    bands: usize,
    bytes_per_sample: usize,
    little_endian: bool,
) -> AnyResult<()> {
    let row_bytes = tile_width * bands * bytes_per_sample;
    match predictor {
        1 => Ok(()),
        2 => {
            if !matches!(bytes_per_sample, 1 | 2 | 4 | 8) {
                return Err(format!("Predictor 2 not supported for {bytes_per_sample}-byte samples").into());
            }
            if row_bytes == 0 {
                return Ok(());
            }
            for row in data.chunks_mut(row_bytes) {
                undo_differencing(row, bands, bytes_per_sample, little_endian);
            }
            Ok(())
        }
        3 => {
            if !matches!(bytes_per_sample, 2 | 4 | 8) {
                return Err(format!("Predictor 3 not supported for {bytes_per_sample}-byte samples").into());
            }
            if row_bytes == 0 {
                return Ok(());
            }
            // Each row holds its samples' most significant bytes, then the next bytes, and so on,
            // differenced byte by byte with a stride of one pixel.
            let mut planes = vec![0; row_bytes];
            for row in data.chunks_exact_mut(row_bytes) {
                running_sum_bytes(row, bands);
                planes.copy_from_slice(row);
                match bytes_per_sample {
                    2 => unshuffle::<2>(&planes, row, little_endian),
                    4 => unshuffle::<4>(&planes, row, little_endian),
                    _ => unshuffle::<8>(&planes, row, little_endian),
                }
            }
            Ok(())
        }
        _ => Err(format!("Unsupported predictor: {predictor}").into()),
    }
}

/// Predictor 2 on one row: every sample becomes the wrapping sum of its band's samples up to it,
/// summed as whole `bytes_per_sample`-byte integers in the file's byte order.
fn undo_differencing(row: &mut [u8], bands: usize, bytes_per_sample: usize, little_endian: bool) {
    macro_rules! words {
        ($t:ty) => {
            if little_endian {
                running_sum(row, bands, |a, b| <$t>::from_le_bytes(a).wrapping_add(<$t>::from_le_bytes(b)).to_le_bytes())
            } else {
                running_sum(row, bands, |a, b| <$t>::from_be_bytes(a).wrapping_add(<$t>::from_be_bytes(b)).to_be_bytes())
            }
        };
    }
    match bytes_per_sample {
        1 => running_sum_bytes(row, bands),
        2 => words!(u16),
        4 => words!(u32),
        _ => words!(u64),
    }
}

/// Every byte becomes the wrapping sum of its band's bytes up to it (`bands` bytes per pixel).
fn running_sum_bytes(row: &mut [u8], bands: usize) {
    if bands == 1 {
        prefix_sum_bytes(row);
    } else {
        running_sum(row, bands, |[a], [b]| [a.wrapping_add(b)]);
    }
}

/// Every `N`-byte sample becomes `add` over its band's samples up to it, one band at a time so
/// that the sum stays in a register.
#[inline]
fn running_sum<const N: usize>(row: &mut [u8], bands: usize, add: impl Fn([u8; N], [u8; N]) -> [u8; N]) {
    let (samples, _) = row.as_chunks_mut::<N>();
    for band in 0..bands {
        let mut sum = [0; N];
        for sample in samples.iter_mut().skip(band).step_by(bands) {
            sum = add(sum, *sample);
            *sample = sum;
        }
    }
}

/// Every byte becomes the wrapping sum of all bytes up to it.
fn prefix_sum_bytes(row: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is part of the x86_64 baseline.
    let (row, mut sum) = unsafe { sse2::prefix_sum_blocks(row) };
    #[cfg(not(target_arch = "x86_64"))]
    let mut sum = 0u8;
    for byte in row {
        sum = sum.wrapping_add(*byte);
        *byte = sum;
    }
}

/// Interleave the `N` byte planes of `planes` (most significant first) into `N`-byte samples in
/// `samples`, in little- or big-endian byte order.
fn unshuffle<const N: usize>(planes: &[u8], samples: &mut [u8], little_endian: bool) {
    let n = planes.len() / N;
    // Planes in the order their bytes take inside a sample
    let mut planes: [&[u8]; N] = std::array::from_fn(|k| &planes[k * n..][..n]);
    if little_endian {
        planes.reverse();
    }
    let (samples, _) = samples.as_chunks_mut::<N>();
    #[cfg(target_arch = "x86_64")]
    let done = match <&[&[u8]; 4]>::try_from(&planes[..]) {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        Ok(&four) => unsafe { sse2::interleave4(four, samples.as_flattened_mut()) },
        Err(_) => 0,
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    for (i, sample) in samples.iter_mut().enumerate().skip(done) {
        for (byte, plane) in sample.iter_mut().zip(&planes) {
            *byte = plane[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi8, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_setzero_si128, _mm_shuffle_epi32,
        _mm_shufflehi_epi16, _mm_slli_si128, _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi8,
        _mm_unpacklo_epi16, _mm_unpacklo_epi8,
    };

    #[target_feature(enable = "sse2")]
    fn load(bytes: &[u8; 16]) -> __m128i {
        // SAFETY: reads the 16 bytes of `bytes`; the load has no alignment requirement.
        unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
    }

    #[target_feature(enable = "sse2")]
    fn store(bytes: &mut [u8; 16], v: __m128i) {
        // SAFETY: writes the 16 bytes of `bytes`; the store has no alignment requirement.
        unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), v) }
    }

    /// [`super::prefix_sum_bytes`] over the whole 16-byte blocks of `row`: returns the bytes
    /// after them and the sum so far.
    #[target_feature(enable = "sse2")]
    pub(super) fn prefix_sum_blocks(row: &mut [u8]) -> (&mut [u8], u8) {
        let (blocks, rest) = row.as_chunks_mut::<16>();
        // The sum of everything before the block, in every lane
        let mut carry = _mm_setzero_si128();
        for block in blocks {
            // Prefix sum inside the block in four shifted adds, then the carry
            let mut v = load(block);
            v = _mm_add_epi8(v, _mm_slli_si128::<1>(v));
            v = _mm_add_epi8(v, _mm_slli_si128::<2>(v));
            v = _mm_add_epi8(v, _mm_slli_si128::<4>(v));
            v = _mm_add_epi8(v, _mm_slli_si128::<8>(v));
            v = _mm_add_epi8(v, carry);
            store(block, v);
            // Broadcast byte 15: to word 7, word 7 to dword 3, dword 3 to all
            carry = _mm_shuffle_epi32::<0xFF>(_mm_shufflehi_epi16::<0xFF>(_mm_unpackhi_epi8(v, v)));
        }
        (rest, _mm_cvtsi128_si32(carry).to_le_bytes()[0])
    }

    /// [`super::unshuffle`] of four planes over whole blocks of 16 samples: writes the samples
    /// `[a[i], b[i], c[i], d[i]]` to `samples` and returns how many it wrote.
    #[target_feature(enable = "sse2")]
    pub(super) fn interleave4([a, b, c, d]: [&[u8]; 4], samples: &mut [u8]) -> usize {
        let (out, _) = samples.as_chunks_mut::<64>();
        let mut done = 0;
        for ((((out, a), b), c), d) in
            out.iter_mut().zip(a.as_chunks::<16>().0).zip(b.as_chunks::<16>().0).zip(c.as_chunks::<16>().0).zip(d.as_chunks::<16>().0)
        {
            let (a, b, c, d) = (load(a), load(b), load(c), load(d));
            // a0 b0 a1 b1 ... and c0 d0 c1 d1 ..., then a0 b0 c0 d0 a1 b1 c1 d1 ...
            let (ab_lo, ab_hi) = (_mm_unpacklo_epi8(a, b), _mm_unpackhi_epi8(a, b));
            let (cd_lo, cd_hi) = (_mm_unpacklo_epi8(c, d), _mm_unpackhi_epi8(c, d));
            let interleaved = [
                _mm_unpacklo_epi16(ab_lo, cd_lo),
                _mm_unpackhi_epi16(ab_lo, cd_lo),
                _mm_unpacklo_epi16(ab_hi, cd_hi),
                _mm_unpackhi_epi16(ab_hi, cd_hi),
            ];
            for (out, v) in out.as_chunks_mut::<16>().0.iter_mut().zip(interleaved) {
                store(out, v);
            }
            done += 16;
        }
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CogReader;
    use std::path::Path;
    use std::process::Command;

    /// Deterministic pseudo-random bytes (xorshift).
    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s.to_le_bytes()[0]
            })
            .collect()
    }

    #[test]
    fn prefix_sum_is_a_running_sum_at_every_length() {
        // lengths around and across the 16-byte blocks: carry between blocks, scalar tail
        for len in 0..100 {
            let data = noise(len, len as u64);
            let mut want = data.clone();
            for i in 1..len {
                want[i] = want[i].wrapping_add(want[i - 1]);
            }
            let mut got = data;
            prefix_sum_bytes(&mut got);
            assert_eq!(got, want, "length {len}");
        }
    }

    #[test]
    fn unshuffle_interleaves_four_planes_in_either_byte_order() {
        // sample counts around and across the 16-sample blocks
        for n in 0..70 {
            let planes = noise(4 * n, n as u64);
            for little_endian in [false, true] {
                let mut got = vec![0; 4 * n];
                unshuffle::<4>(&planes, &mut got, little_endian);
                for i in 0..n {
                    for k in 0..4 {
                        let at = if little_endian { 3 - k } else { k };
                        assert_eq!(got[4 * i + at], planes[k * n + i], "{n} samples, sample {i}, plane {k}, LE {little_endian}");
                    }
                }
            }
        }
    }

    /// Every tile of `path` through [`CogReader`] equals GDAL's reading, bit for bit.
    fn assert_tiles_match_gdal(path: &Path, label: &str) {
        let reader = CogReader::open(path.to_str().unwrap()).unwrap();
        let m = &reader.metadata;
        let ds = gdal::Dataset::open(path).unwrap();
        let bands: Vec<Vec<f32>> = (1..=m.bands)
            .map(|b| {
                let buf: gdal::raster::Buffer<f32> =
                    ds.rasterband(b).unwrap().read_as((0, 0), (m.width, m.height), (m.width, m.height), None).unwrap();
                buf.data().to_vec()
            })
            .collect();
        for ty in 0..m.tiles_down {
            for tx in 0..m.tiles_across {
                let tile = reader.read_tile(ty * m.tiles_across + tx).unwrap();
                for y in 0..m.tile_height.min(m.height - ty * m.tile_height) {
                    for x in 0..m.tile_width.min(m.width - tx * m.tile_width) {
                        for (b, band) in bands.iter().enumerate() {
                            let ours = tile[(y * m.tile_width + x) * m.bands + b];
                            let gdal = band[(ty * m.tile_height + y) * m.width + tx * m.tile_width + x];
                            assert_eq!(ours.to_bits(), gdal.to_bits(), "{label}: tile ({tx}, {ty}) pixel ({x}, {y}) band {b}: {ours} vs {gdal}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn predictor_tiles_decode_like_gdal_in_both_byte_orders() {
        let rgb = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/natural_earth_rgb.tif");
        let dem = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/copernicus_dem_san_francisco.tif");
        for fixture in [rgb, dem] {
            if !Path::new(fixture).exists() {
                println!("Skipping: {fixture} not found");
                return;
            }
        }
        if !Command::new("gdal_translate").arg("--version").output().is_ok_and(|o| o.status.success()) {
            println!("Skipping: gdal_translate not available");
            return;
        }
        let cases: &[(&str, &str, &[&str], u16)] = &[
            ("1-band u8", rgb, &["-b", "1"], 2),
            ("3-band u8", rgb, &[], 2),
            ("u16", rgb, &["-b", "1", "-ot", "UInt16", "-scale", "0", "255", "0", "65535"], 2),
            ("3-band u16", rgb, &["-ot", "UInt16", "-scale", "0", "255", "0", "65535"], 2),
            ("i16", dem, &["-ot", "Int16"], 2),
            ("i32", dem, &["-ot", "Int32", "-scale", "-100", "1000", "-2000000000", "2000000000"], 2),
            ("f32", dem, &[], 2),
            ("f64", dem, &["-ot", "Float64"], 2),
            ("f32", dem, &[], 3),
            ("f32 from bytes", rgb, &["-b", "1", "-ot", "Float32"], 3),
            ("3-band f32", rgb, &["-ot", "Float32"], 3),
            ("f64", dem, &["-ot", "Float64"], 3),
        ];
        let dir = tempfile::tempdir().unwrap();
        for &(name, src, args, predictor) in cases {
            // 300 x 200 in 128 x 128 tiles: partial tiles on the right and bottom edges
            let window = if src == rgb { ["-srcwin", "0", "0", "300", "200"] } else { ["-srcwin", "1500", "1500", "300", "200"] };
            for endianness in ["LITTLE", "BIG"] {
                let label = format!("{name}, predictor {predictor}, {endianness}-endian");
                let out = dir.path().join(format!("{}.tif", label.replace([' ', ','], "_")));
                let status = Command::new("gdal_translate")
                    .arg("-q")
                    .args(window)
                    .args(args)
                    .args(["-co", "TILED=YES", "-co", "BLOCKXSIZE=128", "-co", "BLOCKYSIZE=128", "-co", "COMPRESS=DEFLATE"])
                    .args(["-co", &format!("PREDICTOR={predictor}"), "-co", &format!("ENDIANNESS={endianness}")])
                    .arg(src)
                    .arg(&out)
                    .status();
                assert!(status.is_ok_and(|s| s.success()), "{label}: gdal_translate failed");
                // ENDIANNESS is honoured, not dropped with a warning
                let magic = if endianness == "BIG" { b"MM" } else { b"II" };
                assert_eq!(&std::fs::read(&out).unwrap()[..2], magic, "{label}");
                assert_tiles_match_gdal(&out, &label);
            }
        }
    }
}
