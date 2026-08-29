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

Results are written to ./output/sa/<pack id>/<burst>/ - relative to where the
command is run, not into the data tree.

numpy is required. OpenCV is used when present (faster warps and PNG I/O) but
the script falls back to numpy-only paths, so it runs on a bare Thunder too.
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
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
def to_u8(img: np.ndarray, lo_pct=1.0, hi_pct=99.0, lo=None, hi=None,
          return_range=False):
    """Map to 8 bits, either on this image's own percentiles or a given range.

    A sequence must share one range: recomputing it per frame ties the display
    to the scene's own brightness and the video flickers, which is precisely
    what an SA result is meant to let you judge.
    """
    finite = np.isfinite(img)
    if not finite.any():
        out = np.zeros(img.shape, np.uint8)
        return (out, 0.0, 1.0) if return_range else out
    if lo is None or hi is None:
        lo, hi = (float(x) for x in np.percentile(img[finite], [lo_pct, hi_pct]))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    out[~finite] = 0
    out = (out * 255).astype(np.uint8)
    return (out, lo, hi) if return_range else out


def human_time(sec: float):
    sec = max(0.0, sec)
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    return f"{sec:.0f}s"


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
def parse_frames(spec, n_frames):
    """"all" | "45" | "10-70" | "10:70:5" | "1,5,9" -> list of frame indices."""
    if spec is None:
        return [n_frames // 2]
    spec = spec.strip()
    if spec == "all":
        return list(range(n_frames))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            a, b, step = (part.split(":") + ["1"])[:3]
            out.extend(range(int(a), int(b) + 1, int(step)))
        elif "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in sorted(set(out)) if 0 <= i < n_frames]


def parse_sweep(spec: str):
    a, b, step = (float(x) for x in spec.split(":"))
    n = int(round((b - a) / step)) + 1
    return [a + k * step for k in range(n)]


def encode_video(png_dir: Path, out_path: Path, fps: float):
    """PNG sequence -> mp4. ffmpeg if it is there, OpenCV otherwise."""
    import shutil as _sh
    import subprocess
    pngs = sorted(png_dir.glob("frame*.png"))
    if len(pngs) < 2:
        return None
    if _sh.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-framerate", f"{fps:g}",
               "-pattern_type", "glob", "-i", str(png_dir / "frame*.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               str(out_path)]
        if subprocess.run(cmd).returncode == 0:
            return out_path
        return None
    if cv2 is None:
        return None
    first = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h), first.ndim == 3)
    for p in pngs:
        vw.write(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
    vw.release()
    return out_path


def label_tile(img_u8: np.ndarray, text: str):
    if cv2 is None:
        return img_u8
    out = img_u8.copy()
    if out.ndim == 2:
        out = np.dstack([out] * 3)
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def mosaic_2x2(tiles):
    """Four labelled tiles into one frame; grey where a band is missing."""
    hs = [t.shape[0] for t in tiles if t is not None]
    ws = [t.shape[1] for t in tiles if t is not None]
    h, w = (max(hs), max(ws)) if hs else (16, 16)
    filled = []
    for t in tiles:
        if t is None:
            t = np.full((h, w, 3), 40, np.uint8)
        elif t.ndim == 2:
            t = np.dstack([t] * 3)
        filled.append(t)
    return np.vstack([np.hstack(filled[:2]), np.hstack(filled[2:])])


def load_all_cameras(cams, idx, scale, pool):
    """One frame from every camera, in parallel - numpy releases the GIL."""
    def one(i):
        path, prefix = cams[i]
        img, n, meta = load_camera_frame(path, prefix, idx)
        if scale != 1.0:
            if cv2 is not None:
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            else:
                step = max(1, int(round(1 / scale)))
                img = img[::step, ::step]
        return i, img, n, meta

    futures = [pool.submit(one, i) for i in sorted(cams)]
    images, metas, n_frames = {}, {}, None
    for fut in futures:
        i, img, n, meta = fut.result()
        images[i], metas[i] = img, meta
        n_frames = n if n_frames is None else min(n_frames, n)
    return images, metas, n_frames


def process_frame(cams, idx, cams_of_band, ref, z, f_px, pitch, scale,
                  ksize, calib, pool):
    """One frame -> {layer name: image}. The whole per-frame pipeline."""
    images, metas, _ = load_all_cameras(cams, idx, scale, pool)
    images, gains = band_gain_normalise(images, cams_of_band)

    def warp(i):
        return i, warp_for_plane(images[i], i, ref, z, f_px, pitch, calib)

    warped, shifts = {}, {}
    for i, (w, sh) in [f.result() for f in [pool.submit(warp, i) for i in sorted(images)]]:
        warped[i], shifts[i] = w, sh

    comp, spread, _ = band_composites(warped, cams_of_band)
    layers = {}
    for band, img in comp.items():
        layers[f"band{band}"] = img
    for band, img in spread.items():
        layers[f"spread{band}"] = img
    cross = cross_band_composite(comp, ksize)
    if cross is not None:
        layers["crossband"] = cross
    return layers, comp, gains, shifts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("burst", type=Path,
                    help="packed burst directory, or a merged burstNNNN.npz")
    ap.add_argument("--z", type=float, default=3000.0,
                    help="focal plane distance in mm (default 3000)")
    ap.add_argument("--sweep", default=None,
                    help="pick the plane first: start:stop:step in mm, e.g. 1500:8000:250")
    ap.add_argument("--frames", default=None,
                    help="which frames of the burst: all | 45 | 10-70 | 10:70:5 "
                         "(default: the middle one)")
    ap.add_argument("--frame", type=int, default=None, help="alias for --frames N")
    ap.add_argument("--fps", type=float, default=15.0, help="video frame rate (default 15)")
    ap.add_argument("--no-video", action="store_true", help="write the PNG sequences only")
    ap.add_argument("--drop-png", action="store_true",
                    help="delete each PNG sequence once its video is written "
                         "(a 90-frame burst is ~2 GB of PNG against ~350 MB of video)")
    ap.add_argument("--save-npy", action="store_true",
                    help="also keep the float data (6 MB per layer per frame)")
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
    ap.add_argument("--stretch", choices=("fixed", "per-frame"), default="fixed",
                    help="8-bit mapping: fixed over the sequence (default, no flicker) "
                         "or recomputed per frame")
    ap.add_argument("--workers", type=int, default=0,
                    help="threads for the per-camera work (default: cameras present)")
    ap.add_argument("--calib", type=Path, default=None,
                    help="calibration_results.npz -> homography warp (hook, not implemented)")
    ap.add_argument("--cameras", default=None, help="restrict, e.g. e00,e01,e08,e09")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: ./output/sa/<pack id>/<burst>)")
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

    workers = args.workers or max(1, len(cams))
    pool = ThreadPoolExecutor(max_workers=workers)

    # how many frames the burst holds, and which of them to do
    probe_path, probe_prefix = cams[sorted(cams)[0]]
    _, n_frames, _ = load_camera_frame(probe_path, probe_prefix, 0)
    spec = args.frames if args.frames is not None else (
        str(args.frame) if args.frame is not None else None)
    frames = parse_frames(spec, n_frames)
    if not frames:
        raise SystemExit(f"no frame selected out of {n_frames}")

    ref = int(args.ref[1:]) if args.ref else min(
        cams, key=lambda i: (cam_rc(i)[0] - 1.5) ** 2 + (cam_rc(i)[1] - 1.5) ** 2)
    f_px = args.fpx * (args.scale if args.scale != 1.0 else 1.0)
    print(f"frames  : {len(frames)} of {n_frames} "
          f"({frames[0]}..{frames[-1]})   {workers} worker(s)")
    print(f"ref cam : {cam_name(ref)} ({cam_band(ref)}nm)   pitch {args.pitch}mm   "
          f"f {f_px:.0f}px")

    pack_id = args.burst.parent.name
    out_dir = args.out or Path.cwd() / "output" / "sa" / pack_id / label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output  : {out_dir}")

    # ---- pick the plane once, on the middle frame of the selection --------
    costs = []
    z = args.z
    if args.sweep:
        probe = frames[len(frames) // 2]
        images, _, _ = load_all_cameras(cams, probe, args.scale, pool)
        images, _ = band_gain_normalise(images, cams_of_band)
        for zz in parse_sweep(args.sweep):
            warped = {i: warp_for_plane(images[i], i, ref, zz, f_px,
                                        args.pitch, args.calib)[0] for i in images}
            c = focus_cost(warped, cams_of_band)
            costs.append((zz, c))
            print(f"  z={zz:8.1f}mm  within-band IQR {c:10.3f}")
        z = min(costs, key=lambda zc: zc[1])[0]
        print(f"plane   : z = {z:.1f} mm (band-internal cost, frame {probe})")
        (out_dir / "cost.csv").write_text(
            "z_mm,within_band_iqr\n" + "".join(f"{a},{b}\n" for a, b in costs))

    # ---- the sequence ----------------------------------------------------
    # One 8-bit mapping for the whole sequence: recomputing it per frame makes
    # the video flicker with the scene's own brightness, which is exactly the
    # thing an SA result is meant to let you judge.
    stretch = {}
    written = {}
    t0 = time.time()
    for n, idx in enumerate(frames, 1):
        layers, comp, gains, shifts = process_frame(
            cams, idx, cams_of_band, ref, z, f_px, args.pitch, args.scale,
            args.ksize, args.calib, pool)
        fc = false_colour(comp)
        if fc is not None:
            layers["falsecolour"] = fc

        for name, img in layers.items():
            if name == "falsecolour":
                u8 = img                                  # already 8-bit, per channel
            else:
                if args.stretch == "fixed" and name in stretch:
                    lo, hi = stretch[name]
                    u8 = to_u8(img, lo=lo, hi=hi)
                else:
                    u8, lo, hi = to_u8(img, return_range=True)
                    stretch.setdefault(name, (lo, hi))
            d = out_dir / "png" / name
            d.mkdir(parents=True, exist_ok=True)
            write_png(d / f"frame{idx:04d}.png", u8)
            written[name] = written.get(name, 0) + 1
            if args.save_npy and name != "falsecolour":
                dn = out_dir / "npy" / name
                dn.mkdir(parents=True, exist_ok=True)
                np.save(dn / f"frame{idx:04d}.npy", img)

        # a 2x2 of the bands, so one video shows all four at once
        tiles = []
        for band in BANDS:
            key = f"band{band}"
            if key in layers:
                lo, hi = stretch.get(key, (None, None))
                tiles.append(label_tile(to_u8(layers[key], lo=lo, hi=hi), f"{band}nm"))
            else:
                tiles.append(None)
        d = out_dir / "png" / "bands_mosaic"
        d.mkdir(parents=True, exist_ok=True)
        write_png(d / f"frame{idx:04d}.png", mosaic_2x2(tiles))
        written["bands_mosaic"] = written.get("bands_mosaic", 0) + 1

        if n == 1 or n % 10 == 0 or n == len(frames):
            el = time.time() - t0
            print(f"  [{n}/{len(frames)}] frame {idx:04d}   "
                  f"{el / n:.2f}s/frame   eta {human_time((len(frames) - n) * el / n)}")

    # ---- videos ----------------------------------------------------------
    videos = []
    if not args.no_video and len(frames) > 1:
        (out_dir / "video").mkdir(exist_ok=True)
        for name in sorted(written):
            seq = out_dir / "png" / name
            v = encode_video(seq, out_dir / "video" / f"{name}.mp4", args.fps)
            if v:
                videos.append(v.name)
                if args.drop_png:
                    for f in seq.glob("frame*.png"):
                        f.unlink()
                    seq.rmdir()
        print(f"videos  : {', '.join(videos) if videos else '(none written)'}")

    (out_dir / f"{label}_sa.json").write_text(json.dumps({
        "burst": label,
        "pack_id": pack_id,
        "cameras": [cam_name(i) for i in sorted(cams)],
        "bands": {str(b): [cam_name(i) for i in cams_of_band[b]] for b in cams_of_band},
        "frames": frames,
        "n_frames_in_burst": n_frames,
        "z_mm": z,
        "sweep": [{"z_mm": a, "within_band_iqr": b} for a, b in costs],
        "reference_camera": cam_name(ref),
        "pitch_mm": args.pitch,
        "f_px": f_px,
        "scale": args.scale,
        "fps": args.fps,
        "stretch": {k: [float(v[0]), float(v[1])] for k, v in stretch.items()},
        "stretch_mode": args.stretch,
        "shift_px": {cam_name(i): list(shifts[i]) for i in shifts},
        "band_gain_scale": {cam_name(i): gains[i] for i in gains},
        "layers": {k: v for k, v in sorted(written.items())},
        "videos": videos,
        "note": ("composites and spreads are per band; the cross-band image is "
                 "locally normalised structure, not radiometry"),
    }, indent=2))

    pool.shutdown()
    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\nwrote {sum(written.values())} image(s) in {len(written)} layer(s), "
          f"{total / 2**30:.2f} GiB, to {out_dir}")
    print("  png/band<nm>/          per-band composite (median of that band's views)")
    print("  png/spread<nm>/        within-band IQR - high where an occluder sits")
    print("  png/crossband/         locally normalised structure across bands")
    print("  png/falsecolour/       900/800/750 nm as R/G/B, each stretched alone")
    print("  png/bands_mosaic/      the four bands side by side")
    if videos:
        print("  video/<layer>.mp4      the same sequences as video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
