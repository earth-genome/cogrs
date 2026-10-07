//! Test-only helpers: a tiny deterministic tiled-GeoTIFF (COG) encoder.
//!
//! The crate's [`GeoTiffWriter`](crate::GeoTiffWriter) only writes single-strip files, so tests
//! that need tiling, overviews, sparse tiles, predictors or particular CRSs build their input
//! here. Everything is little-endian classic TIFF with the header, IFDs and tag values placed
//! before the tile data, like a real COG.

use std::io::Write;

/// Sample type of the synthetic raster.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sample {
    U8,
    U16,
    F32,
}

impl Sample {
    fn bytes(self) -> usize {
        match self {
            Sample::U8 => 1,
            Sample::U16 => 2,
            Sample::F32 => 4,
        }
    }
}

/// Description of a synthetic COG. Overview `k` is the nearest-neighbour `2^k` decimation of
/// the full-resolution raster.
#[derive(Clone)]
pub(crate) struct CogSpec {
    pub width: usize,
    pub height: usize,
    pub tile: usize,
    pub bands: usize,
    pub sample: Sample,
    pub deflate: bool,
    /// TIFF predictor 2 (horizontal differencing); `U8`/`U16` only.
    pub predictor: bool,
    pub epsg: u16,
    /// World coordinates of the top-left corner.
    pub origin: (f64, f64),
    pub pixel_size: (f64, f64),
    pub nodata: Option<f64>,
    pub overviews: usize,
    /// `(level, tile_index)` pairs written as sparse tiles (offset 0, byte count 0). Level 0 is
    /// full resolution, level `k >= 1` is overview `k - 1`.
    pub sparse: Vec<(usize, usize)>,
    /// `(level, tile_index)` pairs filled with undecodable bytes.
    pub corrupt: Vec<(usize, usize)>,
    /// Full-resolution value for `(band, x, y)`.
    pub pixel: fn(usize, usize, usize) -> f64,
}

impl CogSpec {
    pub(crate) fn tiles_across(&self, level: usize) -> usize {
        self.level_dims(level).0.div_ceil(self.tile)
    }

    pub(crate) fn tiles_down(&self, level: usize) -> usize {
        self.level_dims(level).1.div_ceil(self.tile)
    }

    pub(crate) fn level_dims(&self, level: usize) -> (usize, usize) {
        let f = 1usize << level;
        (self.width.div_ceil(f), self.height.div_ceil(f))
    }
}

struct Entry {
    tag: u16,
    ftype: u16,
    count: u32,
    data: Vec<u8>,
}

fn shorts(tag: u16, v: &[u16]) -> Entry {
    Entry { tag, ftype: 3, count: v.len() as u32, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

fn longs(tag: u16, v: &[u32]) -> Entry {
    Entry { tag, ftype: 4, count: v.len() as u32, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

fn doubles(tag: u16, v: &[f64]) -> Entry {
    Entry { tag, ftype: 12, count: v.len() as u32, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

fn ascii(tag: u16, s: &str) -> Entry {
    let mut data = s.as_bytes().to_vec();
    data.push(0);
    Entry { tag, ftype: 2, count: data.len() as u32, data }
}

/// Serialize an IFD that starts at `ifd_start`; out-of-line values follow the IFD.
fn encode_ifd(mut entries: Vec<Entry>, ifd_start: u32, next_ifd: u32) -> Vec<u8> {
    entries.sort_by_key(|e| e.tag);
    let table_len = 2 + entries.len() * 12 + 4;
    let mut table = Vec::with_capacity(table_len);
    let mut extra: Vec<u8> = Vec::new();
    table.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for e in &entries {
        table.extend_from_slice(&e.tag.to_le_bytes());
        table.extend_from_slice(&e.ftype.to_le_bytes());
        table.extend_from_slice(&e.count.to_le_bytes());
        if e.data.len() <= 4 {
            let mut inline = e.data.clone();
            inline.resize(4, 0);
            table.extend_from_slice(&inline);
        } else {
            let off = ifd_start as usize + table_len + extra.len();
            table.extend_from_slice(&(off as u32).to_le_bytes());
            extra.extend_from_slice(&e.data);
            if extra.len() % 2 == 1 {
                extra.push(0);
            }
        }
    }
    table.extend_from_slice(&next_ifd.to_le_bytes());
    table.extend_from_slice(&extra);
    table
}

fn encode_samples(spec: &CogSpec, values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * spec.sample.bytes());
    for &v in values {
        match spec.sample {
            Sample::U8 => out.push(v as u8),
            Sample::U16 => out.extend_from_slice(&(v as u16).to_le_bytes()),
            Sample::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
        }
    }
    out
}

fn encode_tile(spec: &CogSpec, level: usize, tile_idx: usize) -> Vec<u8> {
    if spec.sparse.contains(&(level, tile_idx)) {
        return Vec::new();
    }
    if spec.corrupt.contains(&(level, tile_idx)) {
        return vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03];
    }
    let (w, h) = spec.level_dims(level);
    let across = spec.tiles_across(level);
    let (tx, ty) = (tile_idx % across, tile_idx / across);
    let t = spec.tile;
    let step = 1usize << level;
    let mut values = vec![0.0f64; t * t * spec.bands];
    for ly in 0..t {
        for lx in 0..t {
            let (x, y) = (tx * t + lx, ty * t + ly);
            if x < w && y < h {
                for b in 0..spec.bands {
                    values[(ly * t + lx) * spec.bands + b] = (spec.pixel)(b, x * step, y * step);
                }
            }
        }
    }
    let raw = if spec.predictor {
        assert!(spec.sample != Sample::F32, "predictor 2 is not supported for f32 here");
        // Horizontal differencing per row, per band, wrapping in the sample width.
        let modulus = if spec.sample == Sample::U8 { 256.0 } else { 65536.0 };
        for ly in 0..t {
            let row = &mut values[ly * t * spec.bands..(ly + 1) * t * spec.bands];
            for i in (spec.bands..row.len()).rev() {
                row[i] = (row[i] - row[i - spec.bands]).rem_euclid(modulus);
            }
        }
        encode_samples(spec, &values)
    } else {
        encode_samples(spec, &values)
    };
    if spec.deflate {
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&raw).unwrap();
        enc.finish().unwrap()
    } else {
        raw
    }
}

fn ifd_entries(spec: &CogSpec, level: usize, offsets: &[u32], counts: &[u32]) -> Vec<Entry> {
    let (w, h) = spec.level_dims(level);
    let bits = (spec.sample.bytes() * 8) as u16;
    let mut e = vec![
        longs(256, &[w as u32]),
        longs(257, &[h as u32]),
        shorts(258, &vec![bits; spec.bands]),
        shorts(259, &[if spec.deflate { 8 } else { 1 }]),
        shorts(262, &[if spec.bands >= 3 { 2 } else { 1 }]),
        shorts(277, &[spec.bands as u16]),
        shorts(284, &[1]),
        longs(322, &[spec.tile as u32]),
        longs(323, &[spec.tile as u32]),
        longs(324, offsets),
        longs(325, counts),
    ];
    if spec.sample == Sample::F32 {
        e.push(shorts(339, &vec![3; spec.bands]));
    }
    if spec.predictor {
        e.push(shorts(317, &[2]));
    }
    if level == 0 {
        e.push(doubles(33550, &[spec.pixel_size.0, spec.pixel_size.1, 0.0]));
        e.push(doubles(33922, &[0.0, 0.0, 0.0, spec.origin.0, spec.origin.1, 0.0]));
        let (model_type, crs_key) = if spec.epsg == 4326 { (2, 2048) } else { (1, 3072) };
        e.push(shorts(
            34735,
            &[1, 1, 0, 3, 1024, 0, 1, model_type, 1025, 0, 1, 1, crs_key, 0, 1, spec.epsg],
        ));
        if let Some(nd) = spec.nodata {
            e.push(ascii(42113, &format!("{nd}")));
        }
    } else {
        e.push(longs(254, &[1]));
    }
    e
}

/// Encode `spec` as a little-endian classic TIFF.
pub(crate) fn build_cog(spec: &CogSpec) -> Vec<u8> {
    let levels = spec.overviews + 1;
    let tiles: Vec<Vec<Vec<u8>>> = (0..levels)
        .map(|l| (0..spec.tiles_across(l) * spec.tiles_down(l)).map(|i| encode_tile(spec, l, i)).collect())
        .collect();

    // Pass 1: IFD sizes (pointer values do not affect the size).
    let sizes: Vec<usize> = (0..levels)
        .map(|l| {
            let n = tiles[l].len();
            encode_ifd(ifd_entries(spec, l, &vec![0; n], &vec![0; n]), 0, 0).len()
        })
        .collect();
    let mut starts = Vec::with_capacity(levels);
    let mut pos = 8usize;
    for s in &sizes {
        starts.push(pos);
        pos += s;
    }

    // Tile data follows the IFD area, level by level.
    let mut offsets: Vec<Vec<u32>> = Vec::new();
    let mut counts: Vec<Vec<u32>> = Vec::new();
    let mut data_pos = pos;
    for level_tiles in &tiles {
        let mut o = Vec::new();
        let mut c = Vec::new();
        for t in level_tiles {
            if t.is_empty() {
                o.push(0);
                c.push(0);
            } else {
                o.push(data_pos as u32);
                c.push(t.len() as u32);
                data_pos += t.len();
            }
        }
        offsets.push(o);
        counts.push(c);
    }

    let mut out = Vec::with_capacity(data_pos);
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&(starts[0] as u32).to_le_bytes());
    for l in 0..levels {
        let next = if l + 1 < levels { starts[l + 1] as u32 } else { 0 };
        let ifd = encode_ifd(ifd_entries(spec, l, &offsets[l], &counts[l]), starts[l] as u32, next);
        assert_eq!(ifd.len(), sizes[l]);
        out.extend_from_slice(&ifd);
    }
    for level_tiles in &tiles {
        for t in level_tiles {
            out.extend_from_slice(t);
        }
    }
    out
}
