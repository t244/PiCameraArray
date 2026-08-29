#!/usr/bin/env python3
"""
sa_packed.py - synthetic aperture on one packed burst.

Input is a burst produced by construct/pack_bursts.py, i.e. one directory
holding the cameras that have arrived:

    collected_data/packed/<session>/burst0042/e00.npz e01.npz ... manifest.json

(a merged --format npz burst file works too). Cameras that have not been
uploaded are simply absent; everything below is written to work with the
subset that is there.

WHY A PLAIN AVERAGE ACROSS CAMERAS IS WRONG HERE
------------------------------------------------
The 4x4 array carries a 2x2 bandpass tile repeating over the grid:

        col0    col1    col2    col3
  row0  800nm   750nm   800nm   750nm      e00 e01 e02 e03
  row1  900nm   850nm   900nm   850nm      e04 e05 e06 e07
  row2  800nm   750nm   800nm   750nm      e08 e09 e10 e11
  row3  900nm   850nm   900nm   850nm      e12 e13 e14 e15

Classical synthetic aperture leans on photo-consistency: after warping the
views to a plane, a background point should show the *same intensity*
everywhere, so mean/median/variance across views separates background from
occluder. Across bands that is false twice over - each camera integrates a
different 50 nm slice, so the same surface has a different radiance per band,
and filter transmission plus illumination differ per band by up to ~10x. A
16-view mean is therefore not an image of anything; a 16-view variance mostly
measures the filter, not the occluder.

So this script splits the two roles:

  PHOTOMETRY  stays inside one band, where brightness consistency does hold.
              Each band owns 4 cameras that form a 2x2 sub-aperture (twice
              the grid pitch), and the per-band composite is a median over
              exactly those 4 views. Occluder rejection uses within-band
              variance only.
  GEOMETRY    may cross bands, but only through locally normalised images
              (local mean removed, local contrast divided out). That cancels
              any smooth per-band gain a(x)*I + b(x), which is what an unknown
              spectral reflectance looks like at local scale.

Nothing here compares a raw 750 nm number with a raw 900 nm number.

PIPELINE (the structure meant to be extended)
---------------------------------------------
  1 load        one frame per camera, straight out of the packed burst
  2 radiometry  black level, then a per-camera gain matched inside its band
  3 warp        shift-and-add for a fronto-parallel plane at distance z
  4 combine     per-band median + within-band variance, then a cross-band
                composite of locally normalised band images
  5 sweep       optional z sweep scored by a band-internal cost

Deliberately not done here, in order of what to add first:
  * per-camera intrinsics/extrinsics. Without calibration_results.npz the
    warp is a pure translation, which assumes the cameras are coplanar,
    parallel and identical. Pass --calib to swap in the plane-induced
    homography (hook provided, see warp_for_plane).
  * per-frame temporal alignment. A burst is a time series of a moving
    subject and the Pi clocks disagree; analyze/temporal_align.py estimates
    the integer frame offset per camera. Pass --offsets to apply it.
  * sub-band spectral modelling, hole filling, MPI/NeRF style compositing.
    analyze/multispec_sa.py is the full research pipeline.

USAGE
-----
    python analyze/sa_packed.py collected_data/packed/<session>/burst0042 --list
    python analyze/sa_packed.py collected_data/packed/<session>/burst0042 --z 3000
    python analyze/sa_packed.py .../burst0042 --sweep 1500:8000:250 --scale 0.5
    python analyze/sa_packed.py .../burst0042 --z 3000 --frame 45 --out sa_out

numpy is required. OpenCV is used when present (faster warps and PNG I/O) but
the script falls back to numpy-only paths, so it runs on a bare Thunder too.
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:                                   # optional
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from raw10 import unpack_raw10                    # shared with the rest of analyze/
except ImportError:                                   # keep this file self-contained
    def unpack_raw10(packed: np.ndarray, width: int) -> np.ndarray:
        nbytes = (width + 3) // 4 * 5
        b = packed[..., :nbytes].astype(np.uint16)
        b0, b1, b2, b3, b4 = (b[..., i::5] for i in range(5))
        out = np.stack([(b0 << 2) | (b4 & 3), (b1 << 2) | ((b4 >> 2) & 3),
                        (b2 << 2) | ((b4 >> 4) & 3), (b3 << 2) | ((b4 >> 6) & 3)],
                       axis=-1)
        return out.reshape(*packed.shape[:-1], -1)[..., :width]

N_CAM = 16
GRID = 4
# the repeating 2x2 tile, by (row % 2, col % 2)
BAND_TILE = {(0, 0): 800, (0, 1): 750, (1, 0): 900, (1, 1): 850}
BANDS = sorted(set(BAND_TILE.values()))               # [750, 800, 850, 900]


def cam_name(i: int) -> str:
    return f"e{i:02d}"


def cam_rc(i: int):
    return divmod(i, GRID)


def cam_band(i: int) -> int:
    r, c = cam_rc(i)
    return BAND_TILE[(r % 2, c % 2)]


def cam_xy_mm(i: int, pitch: float):
    """Camera centre on the aperture plane, millimetres, origin at e00."""
    r, c = cam_rc(i)
    return c * pitch, r * pitch


# ----------------------------------------------------------------- loading ---
def _npy_header(fh):
    magic = fh.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not a .npy stream")
    major, _minor = fh.read(2)
    hlen = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
    hdr = ast.literal_eval(fh.read(hlen).decode("latin1").strip())
    return hdr["shape"], np.dtype(hdr["descr"]), hdr["fortran_order"]


def read_one_frame(npz_path: Path, member: str, index: int):
    """Read a single frame out of an npz without materialising the whole array.

    A burst holds 90 frames of 1088x1824 bytes; loading all of them for every
    camera would be 2.6 GB for one burst. np.savez stores members uncompressed,
    so the wanted frame is a plain byte range and can be seeked to.
    """
    with zipfile.ZipFile(npz_path) as z:
        info = z.getinfo(member)
        with z.open(info) as fh:
            shape, dtype, fortran = _npy_header(fh)
            if fortran or len(shape) != 3:
                raise ValueError(f"unexpected layout {shape} in {member}")
            n, h, w = shape
            if not -n <= index < n:
                raise IndexError(f"frame {index} out of range (0..{n - 1})")
            index %= n
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("compressed member")   # caller falls back
            fh.seek(index * h * w * dtype.itemsize, 1)
            buf = fh.read(h * w * dtype.itemsize)
    return np.frombuffer(buf, dtype=dtype).reshape(h, w), n


def load_camera_frame(path: Path, prefix: str, index: int):
    """One camera's frame -> (float32 image, n_frames, meta).

    prefix is "" for a per-camera npz and "e07_" inside a merged burst npz.
    """
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
    raw_member = f"{prefix}frames_raw.npy"
    meta = {}

    def small(name, default=None):
        m = f"{prefix}{name}.npy"
        if m not in names:
            return default
        with zipfile.ZipFile(path) as z, z.open(m) as fh:
            shape, dtype, _ = _npy_header(fh)
            arr = np.frombuffer(fh.read(), dtype=dtype)
        return arr.reshape(shape) if shape else arr[0]

    if raw_member in names:
        h, w = (int(x) for x in np.atleast_1d(small("raw_shape")))
        try:
            packed, n = read_one_frame(path, raw_member, index)
        except (ValueError, KeyError):
            d = np.load(path)                          # fallback: whole member
            arr = d[f"{prefix}frames_raw"]
            n = len(arr)
            packed = arr[index % n]
        img = unpack_raw10(packed, w).astype(np.float32)
        bl = small("black_levels", np.zeros(1))
        black10 = float(np.atleast_1d(bl).flat[0]) / 64.0
        img = np.maximum(img - black10, 0.0)
        meta["black_level10"] = black10
        meta["gain"] = float(small("analogue_gain", 0.0) or 0.0)
    else:                                              # legacy 8-bit Y frames
        try:
            frame, n = read_one_frame(path, f"{prefix}frames.npy", index)
        except (ValueError, KeyError):
            d = np.load(path)
            arr = d[f"{prefix}frames"]
            n = len(arr)
            frame = arr[index % n]
        img = frame.astype(np.float32)
    ts = small("timestamps")
    if ts is not None and getattr(ts, "size", 0):
        meta["timestamp"] = str(np.atleast_1d(ts)[index % len(np.atleast_1d(ts))])
    return img, n, meta


def discover_burst(target: Path):
    """-> (label, {camera index: (path, member prefix)}) for a dir or merged npz."""
    if target.is_dir():
        cams = {}
        for i in range(N_CAM):
            p = target / f"{cam_name(i)}.npz"
            if p.exists():
                cams[i] = (p, "")
        return target.name, cams
    if target.suffix == ".npz":
        with zipfile.ZipFile(target) as z:
            present = {n.split("_")[0] for n in z.namelist()}
        cams = {i: (target, f"{cam_name(i)}_") for i in range(N_CAM)
                if cam_name(i) in present}
        return target.stem, cams
    raise SystemExit(f"not a packed burst: {target}")


# ------------------------------------------------------------- radiometry ---
def band_gain_normalise(images: dict, cams_of_band: dict, percentile=90.0):
    """Match camera gains INSIDE each band; never across bands.

    Two cameras of the same band see the same spectrum, so a single scalar per
    camera is enough to make them comparable - that is all a median across the
    band's views needs. Doing this across bands would fabricate a spectral
    correction the data cannot support.
    """
    scaled, scales = {}, {}
    for band, idxs in cams_of_band.items():
        levels = {i: float(np.percentile(images[i], percentile)) for i in idxs}
        ref = float(np.median([v for v in levels.values() if v > 0]) or 1.0)
        for i in idxs:
            s = ref / levels[i] if levels[i] > 0 else 1.0
            scaled[i] = images[i] * s
            scales[i] = s
    return scaled, scales


def local_normalise(img: np.ndarray, ksize: int = 15, clip: float = 6.0):
    """Locally zero-mean, unit-contrast - the only form comparable across bands.

    Removes any smooth a(x)*I + b(x), which is what an unknown per-band
    reflectance factor looks like over a small window, leaving structure.

    Flat patches - the black border, a saturated highlight - have no local
    contrast to divide by, so the quotient there is noise over noise. The
    denominator gets a floor tied to the image's own typical contrast, and the
    result is clipped: past a few sigma the value carries no structure anyway,
    and leaving it unbounded lets a handful of pixels dominate every later
    median and stretch.
    """
    img = img.astype(np.float32)
    # The warp leaves NaN wherever a view has no data, and those must not be
    # allowed to contaminate their neighbourhood: average over the valid
    # samples in the window instead of over the window.
    valid = np.isfinite(img).astype(np.float64)
    filled = np.where(valid > 0, img, 0.0).astype(np.float64)
    n = _box_sum(valid, ksize)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = _box_sum(filled, ksize) / n
        mu2 = _box_sum(filled * filled, ksize) / n
        sigma = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    empty = n < 0.5
    mu = mu.astype(np.float32); sigma = sigma.astype(np.float32)
    mu[empty] = np.nan
    sigma[empty] = np.nan
    finite = np.isfinite(sigma)
    typical = float(np.median(sigma[finite])) if finite.any() else 0.0  # noqa: E501
    floor = max(0.25 * typical, 1e-6)
    # A soft denominator rather than a hard floor: sqrt(s^2 + f^2) damps the
    # patches with no contrast to speak of instead of amplifying their noise,
    # and stays smooth, so the result does not hinge on the last float bit of
    # whichever box filter ran.
    with np.errstate(invalid="ignore"):
        out = (img - mu) / np.sqrt(sigma * sigma + floor * floor)
    return np.clip(out, -clip, clip)


def _box_sum(img: np.ndarray, k: int):
    """Sum over a k x k window, edge-reflected. NaN-free input only.

    float64 throughout, in both backends. The variance below is computed as
    E[x^2] - E[x]^2, and on raw levels of ~10^3 that difference is where
    float32 loses its last digits: the two moments are large and nearly
    equal, so the subtraction cancels most of the precision and a smooth
    surface can come out with a variance of the wrong sign. Sums of 225
    samples are cheap enough not to economise here.

    numpy's "symmetric" is OpenCV's BORDER_REFLECT (edge value repeated);
    numpy's "reflect" is BORDER_REFLECT_101. They must match or the two
    backends disagree along a k/2-wide frame.
    """
    if cv2 is not None:
        return cv2.boxFilter(img.astype(np.float64), cv2.CV_64F, (k, k),
                             normalize=False, borderType=cv2.BORDER_REFLECT)
    r = k // 2
    pad = np.pad(img.astype(np.float64), r, mode="symmetric")
    ii = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), np.float64)
    ii[1:, 1:] = pad.cumsum(0).cumsum(1)
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]


# ------------------------------------------------------------------ warping --
def shift_image(img: np.ndarray, dx: float, dy: float):
    """Sub-pixel translation, the plane-induced warp for an ideal array."""
    if cv2 is not None:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    h, w = img.shape
    ys, xs = np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32)
    sx, sy = xs - dx, ys - dy
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    fx = (sx - x0).astype(np.float32); fy = (sy - y0).astype(np.float32)
    out = np.full((h, w), np.nan, np.float32)
    vx = (x0 >= 0) & (x0 + 1 < w)
    vy = (y0 >= 0) & (y0 + 1 < h)
    if vx.any() and vy.any():
        X0 = x0[vx]; Y0 = y0[vy]
        a = img[np.ix_(Y0, X0)]; b = img[np.ix_(Y0, X0 + 1)]
        c = img[np.ix_(Y0 + 1, X0)]; d = img[np.ix_(Y0 + 1, X0 + 1)]
        FX = fx[vx][None, :]; FY = fy[vy][:, None]
        out[np.ix_(np.where(vy)[0], np.where(vx)[0])] = (
            (a * (1 - FX) + b * FX) * (1 - FY) + (c * (1 - FX) + d * FX) * FY)
    return out


def warp_for_plane(img, cam_i, ref_i, z_mm, f_px, pitch_mm, calib=None):
    """Align camera cam_i onto the reference view for a plane at z_mm.

    With an ideal array (coplanar, parallel, identical lenses) the warp is a
    translation of f * B / z pixels. Real cameras are not that, which is what
    --calib is for: build the plane-induced homography
        H = K_ref (R - t n^T / z) K_i^-1
    from calibration_results.npz and use it here instead. The rest of the
    pipeline does not care which of the two produced the aligned image.
    """
    if calib is not None:
        raise NotImplementedError(
            "homography warp: fill in with calibration_results.npz "
            "(see analyze/multispec_sa.py load_calibration)")
    xi, yi = cam_xy_mm(cam_i, pitch_mm)
    xr, yr = cam_xy_mm(ref_i, pitch_mm)
    dx = f_px * (xi - xr) / z_mm
    dy = f_px * (yi - yr) / z_mm
    return shift_image(img, dx, dy), (dx, dy)


# --------------------------------------------------------------- compositing -
def robust_stats(stack: np.ndarray):
    """Median, IQR and view count across the first axis, ignoring NaN.

    numpy's nanmedian/nanpercentile fall back to a per-pixel Python loop as
    soon as an axis is given and NaNs are present - two minutes for one burst.
    With at most four views per band, sorting the tiny axis with the invalid
    samples pushed to the end and indexing the quantile per pixel does the
    same job in one vectorised pass.
    """
    valid = np.isfinite(stack)
    filled = np.where(valid, stack, np.inf)
    srt = np.sort(filled, axis=0)
    cnt = valid.sum(axis=0)
    denom = np.maximum(cnt - 1, 0).astype(np.float32)

    def quantile(q):
        pos = denom * q
        lo = np.floor(pos).astype(np.intp)
        hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
        frac = (pos - lo).astype(np.float32)
        a = np.take_along_axis(srt, lo[None], axis=0)[0]
        b = np.take_along_axis(srt, hi[None], axis=0)[0]
        with np.errstate(invalid="ignore"):     # inf * 0 where a pixel has no view
            return np.where(cnt > 0, a * (1.0 - frac) + b * frac, np.nan)

    med = quantile(0.5)
    iqr = np.where(cnt > 1, quantile(0.75) - quantile(0.25), np.nan)
    return med, iqr, cnt


def band_composites(warped: dict, cams_of_band: dict):
    """Per band: robust composite and within-band spread.

    Both are photometric statements, so both stay inside one band. The spread
    is the occluder cue: a wire sits at a different depth, lands somewhere
    else in each view, and inflates the spread of the few views it covers.
    """
    comp, spread, nviews = {}, {}, {}
    for band, idxs in cams_of_band.items():
        have = [i for i in idxs if i in warped]
        if not have:
            continue
        med, iqr, _ = robust_stats(np.stack([warped[i] for i in have]))
        comp[band] = med
        spread[band] = iqr                             # IQR: robust, same units
        nviews[band] = len(have)
    return comp, spread, nviews


def cross_band_composite(comp: dict, ksize: int = 15):
    """Combine bands only after local normalisation.

    The result is a structure image, not a radiometric one: it says where the
    scene has edges that survive in every band, which is what the geometry
    needs. Do not read intensities off it.
    """
    if not comp:
        return None
    norm = [local_normalise(v, ksize) for v in comp.values()]
    med, _, _ = robust_stats(np.stack(norm))
    return med


def focus_cost(warped: dict, cams_of_band: dict):
    """How well the plane explains the scene, judged WITHIN bands only.

    Mean over bands of the median within-band IQR. Cross-band variance would
    be dominated by the filters and would minimise at the wrong depth.
    """
    per_band = []
    for band, idxs in cams_of_band.items():
        have = [i for i in idxs if i in warped]
        if len(have) < 2:
            continue
        _, iqr, _ = robust_stats(np.stack([warped[i] for i in have]))
        with np.errstate(invalid="ignore"):
            per_band.append(float(np.nanmedian(iqr)))
    return float(np.mean(per_band)) if per_band else float("nan")


# -------------------------------------------------------------------- output -
def to_u8(img: np.ndarray, lo_pct=1.0, hi_pct=99.0):
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(img[finite], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    out[~finite] = 0
    return (out * 255).astype(np.uint8)


def write_png(path: Path, img_u8: np.ndarray):
    """Grayscale or RGB PNG. Uses OpenCV when available, else pure zlib."""
    if cv2 is not None:
        cv2.imwrite(str(path), img_u8 if img_u8.ndim == 2 else img_u8[..., ::-1])
        return
    if img_u8.ndim == 2:
        h, w = img_u8.shape
        ctype, planes = 0, img_u8
    else:
        h, w, _ = img_u8.shape
        ctype, planes = 2, img_u8
    raw = b"".join(b"\x00" + planes[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def false_colour(comp: dict):
    """R,G,B = 900, 800, 750 nm, each stretched on its own.

    Per-band stretching is the point: the bands are not on a common scale, so
    a shared one would just show which filter passes most light.
    """
    pick = [900, 800, 750]
    if not all(b in comp for b in pick):
        pick = sorted(comp)[:3]
        if len(pick) < 3:
            return None
    return np.dstack([to_u8(comp[b]) for b in pick])


# ----------------------------------------------------------------------- cli -
def parse_sweep(spec: str):
    a, b, step = (float(x) for x in spec.split(":"))
    n = int(round((b - a) / step)) + 1
    return [a + k * step for k in range(n)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("burst", type=Path,
                    help="packed burst directory, or a merged burstNNNN.npz")
    ap.add_argument("--z", type=float, default=3000.0,
                    help="focal plane distance in mm (default 3000)")
    ap.add_argument("--sweep", default=None,
                    help="scan planes instead: start:stop:step in mm, e.g. 1500:8000:250")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index inside the burst (default: the middle one)")
    ap.add_argument("--ref", default=None,
                    help="reference camera (default: the most central one present)")
    ap.add_argument("--pitch", type=float, default=40.0,
                    help="inter-camera spacing in mm (default 40)")
    ap.add_argument("--fpx", type=float, default=6.0e-3 / 3.45e-6,
                    help="focal length in pixels (default 6mm / 3.45um = 1739)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="downscale before processing (0.5 = half size)")
    ap.add_argument("--ksize", type=int, default=15,
                    help="window for the local normalisation (default 15)")
    ap.add_argument("--calib", type=Path, default=None,
                    help="calibration_results.npz -> homography warp (hook, not implemented)")
    ap.add_argument("--cameras", default=None, help="restrict, e.g. e00,e01,e08,e09")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--list", action="store_true", help="show the burst and exit")
    args = ap.parse_args()

    label, cams = discover_burst(args.burst)
    if args.cameras:
        keep = {c.strip() for c in args.cameras.split(",")}
        cams = {i: v for i, v in cams.items() if cam_name(i) in keep}
    if not cams:
        raise SystemExit("no camera npz found in that burst")

    cams_of_band = {}
    for i in sorted(cams):
        cams_of_band.setdefault(cam_band(i), []).append(i)

    print(f"burst   : {label}")
    print(f"cameras : {len(cams)} present, "
          f"{', '.join(cam_name(i) for i in sorted(cams))}")
    print(f"missing : {', '.join(cam_name(i) for i in range(N_CAM) if i not in cams) or '-'}")
    for band in BANDS:
        have = cams_of_band.get(band, [])
        print(f"  {band}nm: {len(have)} view(s)  "
              f"{', '.join(cam_name(i) for i in have) or '- (band unusable)'}")
    if args.list:
        return 0
    thin = [b for b in BANDS if len(cams_of_band.get(b, [])) < 2]
    if thin:
        print(f"note    : {thin} have fewer than 2 views; their composite is a "
              f"single view and their spread is undefined")

    # ---- 1 load ----------------------------------------------------------
    images, nframes, metas = {}, None, {}
    idx = args.frame
    for i in sorted(cams):
        path, prefix = cams[i]
        probe = 0 if idx is None else idx
        img, n, meta = load_camera_frame(path, prefix, probe)
        if idx is None:                                # default: middle of the burst
            idx = n // 2
            if idx != probe:
                img, n, meta = load_camera_frame(path, prefix, idx)
        nframes = n if nframes is None else min(nframes, n)
        if args.scale != 1.0:
            if cv2 is not None:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_AREA)
            else:
                s = max(1, int(round(1 / args.scale)))
                img = img[::s, ::s]
        images[i] = img
        metas[i] = meta
    h, w = next(iter(images.values())).shape
    f_px = args.fpx * (args.scale if args.scale != 1.0 else 1.0)
    print(f"frame   : index {idx} of {nframes}, {w}x{h} px"
          f"{'' if args.scale == 1 else f' (scaled {args.scale})'}")

    # ---- 2 radiometry ----------------------------------------------------
    images, scales = band_gain_normalise(images, cams_of_band)

    ref = None
    if args.ref:
        ref = int(args.ref[1:])
    else:                                              # most central present camera
        ref = min(cams, key=lambda i: (cam_rc(i)[0] - 1.5) ** 2 + (cam_rc(i)[1] - 1.5) ** 2)
    print(f"ref cam : {cam_name(ref)} ({cam_band(ref)}nm)   pitch {args.pitch}mm   "
          f"f {f_px:.0f}px")

    out_dir = args.out or (args.burst if args.burst.is_dir()
                           else args.burst.parent) / f"sa_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 5 sweep (optional) ---------------------------------------------
    planes = parse_sweep(args.sweep) if args.sweep else [args.z]
    costs = []
    for z in planes:
        warped = {}
        for i in sorted(images):
            warped[i], _ = warp_for_plane(images[i], i, ref, z, f_px,
                                          args.pitch, args.calib)
        c = focus_cost(warped, cams_of_band)
        costs.append((z, c))
        if len(planes) > 1:
            print(f"  z={z:8.1f}mm  within-band IQR {c:10.3f}")
    best_z = min(costs, key=lambda zc: zc[1])[0]
    if len(planes) > 1:
        (out_dir / f"{label}_cost.csv").write_text(
            "z_mm,within_band_iqr\n" + "".join(f"{z},{c}\n" for z, c in costs))
        print(f"best plane by band-internal cost: z = {best_z:.1f} mm")

    # ---- 3 warp + 4 combine at the chosen plane --------------------------
    warped, shifts = {}, {}
    for i in sorted(images):
        warped[i], shifts[i] = warp_for_plane(images[i], i, ref, best_z, f_px,
                                              args.pitch, args.calib)
    comp, spread, nviews = band_composites(warped, cams_of_band)
    cross = cross_band_composite(comp, args.ksize)

    for band in sorted(comp):
        write_png(out_dir / f"{label}_z{best_z:.0f}_band{band}.png", to_u8(comp[band]))
        write_png(out_dir / f"{label}_z{best_z:.0f}_band{band}_spread.png",
                  to_u8(spread[band]))
        np.save(out_dir / f"{label}_z{best_z:.0f}_band{band}.npy", comp[band])
    if cross is not None:
        write_png(out_dir / f"{label}_z{best_z:.0f}_crossband.png", to_u8(cross))
        np.save(out_dir / f"{label}_z{best_z:.0f}_crossband.npy", cross)
    fc = false_colour(comp)
    if fc is not None:
        write_png(out_dir / f"{label}_z{best_z:.0f}_falsecolour.png", fc)

    (out_dir / f"{label}_sa.json").write_text(json.dumps({
        "burst": label,
        "cameras": [cam_name(i) for i in sorted(images)],
        "bands": {str(b): [cam_name(i) for i in cams_of_band[b]] for b in cams_of_band},
        "frame_index": idx,
        "z_mm": best_z,
        "sweep": [{"z_mm": z, "within_band_iqr": c} for z, c in costs],
        "reference_camera": cam_name(ref),
        "pitch_mm": args.pitch,
        "f_px": f_px,
        "scale": args.scale,
        "shift_px": {cam_name(i): list(shifts[i]) for i in shifts},
        "band_gain_scale": {cam_name(i): scales[i] for i in scales},
        "note": ("composites and spreads are per band; the cross-band image is "
                 "locally normalised structure, not radiometry"),
    }, indent=2))

    print(f"\nwrote {len(list(out_dir.iterdir()))} file(s) to {out_dir}")
    print("  *_band<nm>.png        per-band composite (median of that band's views)")
    print("  *_band<nm>_spread.png within-band IQR - high where an occluder sits")
    print("  *_crossband.png       locally normalised structure across bands")
    print("  *_falsecolour.png     900/800/750 nm as R/G/B, each stretched alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
