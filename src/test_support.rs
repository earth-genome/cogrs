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

fn ifd_entries(spec: &CogSpec, level: usize, offsets: &[u32], counts: &[u32], point: bool) -> Vec<Entry> {
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
        // PixelIsPoint: the tiepoint is the centre of pixel (0, 0); PixelIsArea: its outer corner.
        let (tx, ty) = if point {
            (spec.origin.0 + 0.5 * spec.pixel_size.0, spec.origin.1 - 0.5 * spec.pixel_size.1)
        } else {
            spec.origin
        };
        e.push(doubles(33922, &[0.0, 0.0, 0.0, tx, ty, 0.0]));
        let (model_type, crs_key) = if spec.epsg == 4326 { (2, 2048) } else { (1, 3072) };
        e.push(shorts(
            34735,
            &[1, 1, 0, 3, 1024, 0, 1, model_type, 1025, 0, 1, if point { 2 } else { 1 }, crs_key, 0, 1, spec.epsg],
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
    build_cog_registered(spec, false)
}

/// Like [`build_cog`], optionally with `PixelIsPoint` raster registration (`spec.origin` stays the
/// outer corner of pixel (0, 0); the tiepoint is written as the pixel's centre).
pub(crate) fn build_cog_registered(spec: &CogSpec, point: bool) -> Vec<u8> {
    let levels = spec.overviews + 1;
    let tiles: Vec<Vec<Vec<u8>>> = (0..levels)
        .map(|l| (0..spec.tiles_across(l) * spec.tiles_down(l)).map(|i| encode_tile(spec, l, i)).collect())
        .collect();

    // Pass 1: IFD sizes (pointer values do not affect the size).
    let sizes: Vec<usize> = (0..levels)
        .map(|l| {
            let n = tiles[l].len();
            encode_ifd(ifd_entries(spec, l, &vec![0; n], &vec![0; n], point), 0, 0).len()
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
        let ifd = encode_ifd(ifd_entries(spec, l, &offsets[l], &counts[l], point), starts[l] as u32, next);
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

// ============================================================================
// Mock async reader
// ============================================================================

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;

use crate::async_io::{AsyncRangeReader, IoOptions};
use crate::tiff_utils::AnyResult;

/// In-memory [`AsyncRangeReader`] that behaves like a remote source: every read takes
/// `latency` (via `tokio::time::sleep`, so it works with paused time), and the calls made, the
/// number in flight and the high-water mark are recorded.
pub(crate) struct MockReader {
    data: Bytes,
    identifier: String,
    latency: Mutex<Duration>,
    options: IoOptions,
    calls: Mutex<Vec<Range<u64>>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    /// Reads whose range overlaps one of these fail.
    fail_ranges: Mutex<Vec<Range<u64>>>,
}

impl MockReader {
    pub(crate) fn new(data: Vec<u8>, identifier: &str, latency: Duration) -> Self {
        Self {
            data: Bytes::from(data),
            identifier: identifier.to_string(),
            latency: Mutex::new(latency),
            options: IoOptions::default(),
            calls: Mutex::default(),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            fail_ranges: Mutex::default(),
        }
    }

    pub(crate) fn with_options(mut self, options: IoOptions) -> Self {
        self.options = options;
        self
    }

    pub(crate) fn fail_on(&self, range: Range<u64>) {
        self.fail_ranges.lock().push(range);
    }

    /// Ranges requested so far, in call order.
    pub(crate) fn calls(&self) -> Vec<Range<u64>> {
        self.calls.lock().clone()
    }

    pub(crate) fn call_count(&self) -> usize {
        self.calls.lock().len()
    }

    pub(crate) fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }

    pub(crate) fn set_latency(&self, latency: Duration) {
        *self.latency.lock() = latency;
    }

    pub(crate) fn reset(&self) {
        self.calls.lock().clear();
        self.max_in_flight.store(0, Ordering::SeqCst);
    }
}

impl AsyncRangeReader for MockReader {
    fn read_range(&self, offset: u64, len: usize) -> BoxFuture<'_, AnyResult<Bytes>> {
        Box::pin(async move {
            let range = offset..offset + len as u64;
            self.calls.lock().push(range.clone());
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            let latency = *self.latency.lock();
            tokio::time::sleep(latency).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            if self.fail_ranges.lock().iter().any(|f| f.start < range.end && range.start < f.end) {
                return Err(format!("injected failure for {range:?}").into());
            }
            if range.end > self.data.len() as u64 {
                return Err(format!("range {range:?} outside mock of {} bytes", self.data.len()).into());
            }
            Ok(self.data.slice(range.start as usize..range.end as usize))
        })
    }

    fn size(&self) -> u64 {
        self.data.len() as u64
    }

    fn identifier(&self) -> &str {
        &self.identifier
    }

    fn is_local(&self) -> bool {
        false
    }

    fn io_options(&self) -> &IoOptions {
        &self.options
    }
}

// ============================================================================
// Local HTTP range server
// ============================================================================

/// One object served by [`ObjectServer`].
#[derive(Clone)]
pub(crate) struct ServedObject {
    pub data: std::sync::Arc<Vec<u8>>,
    /// `ETag` header value as sent (quotes included), if any
    pub etag: Option<String>,
    /// `Last-Modified` as Unix seconds, if any
    pub modified: Option<i64>,
}

impl ServedObject {
    /// An object with the ETag `"abc"` and a fixed `Last-Modified`.
    pub(crate) fn new(data: Vec<u8>) -> Self {
        use chrono::TimeZone;
        Self {
            data: std::sync::Arc::new(data),
            etag: Some("\"abc\"".to_string()),
            modified: Some(chrono::Utc.with_ymd_and_hms(2026, 10, 6, 18, 56, 17).unwrap().timestamp()),
        }
    }

    pub(crate) fn etag(mut self, etag: Option<&str>) -> Self {
        self.etag = etag.map(str::to_string);
        self
    }

    pub(crate) fn modified(mut self, modified: Option<i64>) -> Self {
        self.modified = modified;
        self
    }
}

/// A request as the server saw it.
#[derive(Clone, Debug)]
pub(crate) struct RecordedRequest {
    pub path: String,
    pub range: Option<(usize, usize)>,
    pub if_match: Option<String>,
    pub if_unmodified_since: Option<String>,
    /// Status code the server answered with
    pub status: u16,
}

struct ServerState {
    /// Served for any path without a specific object (`None`: such paths are 404).
    default: Option<ServedObject>,
    objects: std::collections::HashMap<String, ServedObject>,
    requests: Vec<RecordedRequest>,
    delay: Duration,
    /// Evaluate `If-Match` / `If-Unmodified-Since` (a server that ignores them sets this false).
    honor_conditionals: bool,
    /// Answer 503 to requests whose range starts at or after this offset.
    fail_from: Option<usize>,
}

/// Minimal range-capable HTTP/1.1 server on 127.0.0.1 whose objects can be replaced while it
/// runs. It records every request, answers `412` to failed `If-Match` / `If-Unmodified-Since`
/// conditions and `404` for unknown paths, and delays every response by `delay`, which stands in
/// for network latency. Its threads live for the rest of the process.
#[derive(Clone)]
pub(crate) struct ObjectServer {
    base: String,
    state: std::sync::Arc<Mutex<ServerState>>,
    log: std::sync::Arc<Mutex<Vec<String>>>,
}

impl ObjectServer {
    /// Start a server; `default` is served for any path without its own object.
    pub(crate) fn start(default: Option<ServedObject>, delay: Duration) -> Self {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;
        use std::sync::Arc;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = Self {
            base: format!("http://{addr}"),
            state: Arc::new(Mutex::new(ServerState {
                default,
                objects: std::collections::HashMap::new(),
                requests: Vec::new(),
                delay,
                honor_conditionals: true,
                fail_from: None,
            })),
            log: Arc::new(Mutex::new(Vec::new())),
        };
        let shared = server.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let server = shared.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    loop {
                        let mut request_line = String::new();
                        if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                            return;
                        }
                        let path = request_line.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap().to_string();
                        let (mut range, mut if_match, mut if_unmodified_since) = (None, None, None);
                        loop {
                            let mut line = String::new();
                            reader.read_line(&mut line).unwrap();
                            if line.trim().is_empty() {
                                break;
                            }
                            let lower = line.to_ascii_lowercase();
                            if let Some(v) = lower.strip_prefix("range: bytes=") {
                                let (a, b) = v.trim().split_once('-').unwrap();
                                range = Some((a.parse::<usize>().unwrap(), b.parse::<usize>().ok()));
                            } else if lower.starts_with("if-match:") {
                                if_match = Some(line["if-match:".len()..].trim().to_string());
                            } else if lower.starts_with("if-unmodified-since:") {
                                if_unmodified_since = Some(line["if-unmodified-since:".len()..].trim().to_string());
                            }
                        }
                        let requested = range.map(|(a, b)| (a, b.unwrap_or(usize::MAX)));
                        let (response, status, _) =
                            server.respond(&request_line, &path, range, if_match.clone(), if_unmodified_since.clone());
                        server.state.lock().requests.push(RecordedRequest {
                            path,
                            range: requested,
                            if_match,
                            if_unmodified_since,
                            status,
                        });
                        // One write: separate header and body writes stall ~40 ms on Nagle/delayed ACK.
                        if stream.write_all(&response).is_err() {
                            return;
                        }
                    }
                });
            }
        });
        server
    }

    /// Build the response bytes, status and the (resolved) byte range for one request.
    fn respond(
        &self,
        request_line: &str,
        path: &str,
        range: Option<(usize, Option<usize>)>,
        if_match: Option<String>,
        if_unmodified_since: Option<String>,
    ) -> (Vec<u8>, u16, Option<(usize, usize)>) {
        let (object, delay, honor, fail_from) = {
            let state = self.state.lock();
            let object = state.objects.get(path).or(state.default.as_ref()).cloned();
            (object, state.delay, state.honor_conditionals, state.fail_from)
        };
        let resolved = range.map(|(a, b)| (a, b.unwrap_or(usize::MAX)));
        self.log.lock().push(format!("{} range={:?}", request_line.trim(), resolved));
        std::thread::sleep(delay);
        let empty = |status: &str| format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").into_bytes();
        let Some(object) = object else {
            return (empty("404 Not Found"), 404, None);
        };
        if fail_from.is_some_and(|from| resolved.map_or(0, |(a, _)| a) >= from) {
            return (empty("503 Service Unavailable"), 503, None);
        }
        if honor && precondition_failed(&object, if_match.as_deref(), if_unmodified_since.as_deref()) {
            return (empty("412 Precondition Failed"), 412, None);
        }
        let last = object.data.len() - 1;
        let (start, end) = resolved.map_or((0, last), |(a, b)| (a, b.min(last)));
        let body = &object.data[start..=end];
        let mut head = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\n",
            body.len(),
            object.data.len()
        );
        if let Some(etag) = &object.etag {
            head.push_str(&format!("ETag: {etag}\r\n"));
        }
        if let Some(modified) = object.modified {
            let date = chrono::DateTime::from_timestamp(modified, 0).unwrap();
            head.push_str(&format!("Last-Modified: {}\r\n", date.format("%a, %d %b %Y %H:%M:%S GMT")));
        }
        head.push_str("\r\n");
        let mut response = head.into_bytes();
        response.extend_from_slice(body);
        (response, 206, Some((start, end)))
    }

    /// Base URL (`http://127.0.0.1:port`).
    pub(crate) fn base(&self) -> &str {
        &self.base
    }

    /// Replace the object served for paths without their own object.
    pub(crate) fn replace_default(&self, object: Option<ServedObject>) {
        self.state.lock().default = object;
    }

    pub(crate) fn set_honor_conditionals(&self, honor: bool) {
        self.state.lock().honor_conditionals = honor;
    }

    pub(crate) fn requests(&self) -> Vec<RecordedRequest> {
        self.state.lock().requests.clone()
    }

    /// Request lines (`GET /path HTTP/1.1 range=Some((a, b))`), as [`serve_bytes`] returns them.
    pub(crate) fn log(&self) -> std::sync::Arc<Mutex<Vec<String>>> {
        std::sync::Arc::clone(&self.log)
    }
}

/// Whether an `If-Match` / `If-Unmodified-Since` condition fails for `object`.
fn precondition_failed(object: &ServedObject, if_match: Option<&str>, if_unmodified_since: Option<&str>) -> bool {
    if let Some(tags) = if_match {
        // Strong comparison, as RFC 9110 specifies for If-Match.
        let matches = tags.trim() == "*"
            || tags.split(',').map(str::trim).any(|t| !t.starts_with("W/") && object.etag.as_deref() == Some(t));
        if !matches {
            return true;
        }
    }
    if let (Some(date), Some(modified)) = (if_unmodified_since, object.modified)
        && let Ok(since) = chrono::DateTime::parse_from_rfc2822(date)
        && modified > since.timestamp()
    {
        return true;
    }
    false
}

/// Minimal range-capable HTTP/1.1 server over a byte buffer on 127.0.0.1.
///
/// Returns its base URL (`http://127.0.0.1:port`, any path is served) and a log of request
/// lines. Every response is delayed by `delay`, which stands in for network latency.
pub(crate) fn serve_bytes(data: Vec<u8>, delay: Duration) -> (String, std::sync::Arc<Mutex<Vec<String>>>) {
    let server = ObjectServer::start(Some(ServedObject::new(data)), delay);
    (server.base().to_string(), server.log())
}
