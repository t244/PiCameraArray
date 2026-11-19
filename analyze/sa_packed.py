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
              the grid pitch), and the per-band composite combines exactly
              those 4 views - weighted by how well each agrees over a
              WINDOW, not by a per-pixel statistic, because occlusion is
              spatially coherent: a bar covers a neighbourhood of a view,
              not a sample.
  GEOMETRY    may cross bands, but only through locally normalised images
              (local mean removed, local contrast divided out). That cancels
              any smooth per-band gain a(x)*I + b(x), which is what an unknown
              spectral reflectance looks like at local scale.

Nothing here compares a raw 750 nm number with a raw 900 nm number.

PIPELINE (the structure meant to be extended)
---------------------------------------------
  1 load        one frame per camera, straight out of the packed burst
  2 radiometry  black level, then a per-camera gain matched inside its band
  3 warp        onto a fronto-parallel plane at distance z: the nominal
                shift-and-add, or the measured homography with --calib
  4 combine     per-band locally-weighted composite + the occlusion fraction,
                then a cross-band composite of locally normalised band images
  5 sweep       optional z sweep scored by a band-internal cost
  6 refine      optional --refine: the residual per-camera shift the images
                themselves show, on top of the calibration

CALIBRATION
-----------
Without --calib the warp is a pure translation, which takes the cameras to be
coplanar, parallel and identical. They are not: calibration_results.npz has
them 0.4 to 3.3 degrees off each other, which at f = 1800 px is up to a
hundred pixels across the frame that no translation can remove - and a
misaligned view does not just blur the composite, it inflates the within-band
spread everywhere and buries the occluder cue the spread exists to show.

    python analyze/calibrate.py <charuco images>     # writes the npz
    python analyze/sa_packed.py <burst> --calib calibration_results.npz

With it, each view is resampled by the plane-induced homography built from the
measured intrinsics and relative poses, lens distortion folded into the same
resample; the output frame is the reference camera seen through an ideal
pinhole. --fpx and --pitch then only serve cameras the file does not cover.

The npz on hand was shot on a different prototype, so treat it as a starting
point and not as the truth. What a pose error leaves after the plane warp is,
to first order, a constant shift per camera, and that is measurable on the
burst: --refine matches every view against the consensus of the others over a
bounded search and applies what it finds. It refuses to move a camera whose
correlation peak is weak, and prints every peak, so a burst that cannot
support the measurement says so instead of inventing an alignment. Use
--refine-ksize at or above the period of whatever structure the views share -
below it the local normalisation deletes the shared structure and leaves only
the view-dependent detail.

THE SETUP THIS IS BUILT FOR
---------------------------
A rotating tumbler whose mesh wall sits about 100 mm from the lens, with 20 mm
soft capsules tumbling 400-700 mm away. Every default follows from those three
numbers (--occluder-mm, --target-mm, --target-size-mm), and so do the limits:

  a capsule spans 65 px at 550 mm, so every local window is a fraction of that
  one plane holds a capsule inside its own width only from 471 to 660 mm, so a
    depth range wider than that needs more than one plane, not a better one
  the fence moves 587 px between neighbouring cameras and smears 1760 px across
    the aperture, so no two views look through the same part of it

That last number is what makes the fence tractable and also what makes the
usual estimators wrong for it. The fence is not an opaque bar that some views
miss: it is defocused to about its own period and roughly half open, so it
attenuates EVERY view, multiplicatively, by a different amount. Contamination
is therefore one-sided - it can only remove light - so the answer is in the
upper half of the views, not in the middle where a median looks. Hence
--combine bright, which ranks the views by their LOCAL LEVEL, drops the darkest
(most occluded) and the brightest (as likely a specular flash off a capsule as
a clear view of one), and averages the whole pixels of what is left. Selecting
views rather than blending them is the other half of it: a median of four
imperfectly aligned views blends the capsule edges away, and on burst0001 that
is the difference between capsules you can count and a grey mush.

What no plane can fix is an occluder at the target's own depth, and what no
aperture can fix is a periodic occluder whose phase happens to be the same in
every view. Both are reported rather than left to be discovered:

  the sweep prints a fence-phase column - 1 means every view sees the fence at
    the same phase, so that plane cannot cancel it however good the estimator
  --notch-fence writes defence<nm>, the composite with the fence's own lattice
    frequencies removed. Symmetry is the only handle left when parallax has
    none, and it is opt-in because it will also erase a periodic subject

READING THE OUTPUT
------------------
band<nm>    the composite. Radiometric within its band.
spread<nm>  the OCCLUSION FRACTION in 0..1: the share of the light that band's
            views lose to something in front, averaged over a --ksize window
            and divided by the local light level. A bar covering k of the V
            views reads about k/V. It is deliberately not an intensity spread
            - in raw units a bright specular glint outranks a dark fence bar,
            which is why the layer used to show the highlights and hide the
            occluder. The 8-bit mapping is fixed at 0..1, never stretched.
gloss<nm>   the same disagreement seen from the other side: light that one view
            has and the others do not. A glossy ellipsoid puts its highlight
            somewhere different for every viewpoint, so this maps the capsules.
            Also 0..1 fixed. In a scene as specular as this one it runs high -
            that is the measurement, not a fault: the upward disagreement here
            is about twice the downward one, i.e. gloss, not occlusion, is what
            makes these views differ.
defence<nm> --notch-fence only: the composite with the fence lattice notched out
capsules    --detect only: target-sized elliptical blobs ringed over the
            composite. Green held its ground when the focus plane was moved
            either side, so it is at this depth; red is target-sized but moved,
            so it is not; blue is a specular highlight. Read the colours, not
            the count - on this rig a 3.5 mm fence cell at 100 mm projects to
            60 px and a 20 mm capsule at 550 mm to 65 px, so SIZE CANNOT TELL
            THEM APART and the depth test is the whole detector. On burst0001
            frame 47 it keeps 1 blob of 55.
crossband   locally normalised structure across bands. Not radiometry.

--gamma and --clahe are display only and are applied to every layer except
spread and gloss, which are absolute fractions and have to stay readable off
the pixel value. --gamma is a fixed curve and is safe for a sequence; --clahe
adapts per frame and will breathe in a video. --gamma 1.6 --clahe 2 reads best
on this scene.

TIMING
------
The Pis answer one trigger but start their bursts a little apart, so frame k
is not the same instant on every camera. analyze/temporal_align.py measures
that as an integer offset per camera and --offsets applies it, after which one
frame index is one instant. Frames that cannot be shifted - the first few when
an offset is negative - are dropped rather than wrapped round to the other end
of the burst, which would be a different instant, not a missing one.

Measured on burst0001, where the offsets run -2..+2: applying them changes the
within-band disagreement by under 1%. Scanning one camera against its
band-mates, with the scene's own drift divided out, e08 does dip at exactly the
-2 it was assigned, but the dip is 4% deep and two of the four bands show no
interior minimum at all. So the timing on that burst is real and small, and it
is NOT why those views fail to agree.

Deliberately not done here, in order of what to add first:
  * sub-band spectral modelling, hole filling, MPI/NeRF style compositing.
    analyze/multispec_sa.py is the full research pipeline.

USAGE
-----
    python analyze/sa_packed.py collected_data/packed/<session>/burst0042 --list
    python analyze/sa_packed.py collected_data/packed/<session>/burst0042 --z 3000
    python analyze/sa_packed.py .../burst0042 --sweep 1500:8000:250 --scale 0.5
    python analyze/sa_packed.py .../burst0042 --z 3000 --frame 45 --out sa_out
    python analyze/sa_packed.py .../burst0042 --calib calibration_results.npz --frames all
    python analyze/sa_packed.py .../burst0042 --calib calibration_results.npz --refine

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
import threading
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


# ------------------------------------------------------------------- scene ---
def scene_scales(f_px, occluder_mm, target_mm, target_size_mm, pitch_mm,
                 near_mm=None, far_mm=None, grid=GRID):
    """Turn the physical setup into the pixel numbers the pipeline needs.

    Every window size, sweep range and step below is a length in the scene,
    not a number of pixels, and the two are related by f/z. Stating the setup
    once and deriving the rest keeps them consistent - and makes the array's
    limits arithmetic rather than something to discover from the images.

    The three that matter for a near occluder in front of a small target:

    target_px       how big the target is on the sensor. Every local window
                    should be a fraction of this, or the composite smears the
                    thing it is meant to show.
    fence_smear_px  how far the occluder travels across the full aperture when
                    the focus is on the target. Bigger than the occluder's own
                    period means the array can average it away; smaller means
                    no estimator can, and the answer is a wider array or a
                    nearer occluder.
    dof_near/far    the planes over which the target still lands within one
                    target width. A capsule outside that range is smeared by
                    more than its own size no matter which plane is chosen, so
                    a depth range wider than this needs more than one plane.
    """
    span = pitch_mm * (grid - 1)
    target_px = target_size_mm * f_px / target_mm
    d = target_px / (f_px * span)                      # the DoF half-width in 1/z
    inv = 1.0 / target_mm
    return {
        "target_px": target_px,
        "target_disparity_px": f_px * pitch_mm / target_mm,
        "fence_disparity_px": f_px * pitch_mm * abs(1.0 / occluder_mm - inv),
        "fence_smear_px": f_px * span * abs(1.0 / occluder_mm - inv),
        "dof_near_mm": 1.0 / (inv + d),
        "dof_far_mm": 1.0 / max(inv - d, 1e-9),
        "sweep_near_mm": near_mm if near_mm else 1.0 / (inv + 2 * d),
        "sweep_far_mm": far_mm if far_mm else 1.0 / max(inv - 2 * d, 1e-9),
        "span_mm": span,
    }


def fence_period_px(img: np.ndarray, lo=8.0, hi=400.0):
    """The occluder's period, read off the image's own horizontal spectrum.

    A woven fence is the strongest periodic thing in the frame, so it stands
    out of a spectrum that is otherwise broadband speckle. Measuring it beats
    deriving it from a nominal mesh pitch, which nobody has to hand, and it is
    what the notch filter and the phase score below both need.

    Returns (period in px, how much of the band it carries) or (nan, 0).
    """
    a = np.nan_to_num(img).astype(np.float64)
    h, w = a.shape
    a = a[h // 6:h - h // 6, w // 6:w - w // 6]
    a = a - a.mean(axis=1, keepdims=True)
    n = a.shape[1]
    # Zero-padded, then a parabolic fit on the peak. A bare bin index is far
    # too coarse: over a 1456 px frame a 5% error in the period is several
    # whole bars, which puts the notch and the phase score on the wrong
    # frequency entirely.
    pad = 4 * n
    sp = np.abs(np.fft.rfft(a * np.hanning(n), n=pad, axis=1)).mean(0)
    k_lo, k_hi = max(2, int(pad / hi)), min(len(sp) - 2, int(pad / lo))
    if k_hi <= k_lo:
        return float("nan"), 0.0
    k = int(np.argmax(sp[k_lo:k_hi])) + k_lo
    y0, y1, y2 = (float(np.log(max(sp[k + d], 1e-30))) for d in (-1, 0, 1))
    den = y0 - 2 * y1 + y2
    k = k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    half = 2 * 4                                       # +-2 unpadded bins
    lo_i, hi_i = int(k - half), int(k + half) + 1
    share = float(sp[lo_i:hi_i].sum() / sp[k_lo:k_hi].sum())
    return pad / k, share


def fence_phase_spread(cams, ref, z_mm, f_px, pitch_mm, occluder_mm, period_px):
    """Can this plane cancel the fence at all? 0 yes, 1 no.

    A periodic occluder is only averaged away when the views see it at
    different phases. After warping to a plane at z the occluder still sits
    where its own parallax puts it, so view i keeps a residual offset

        f * B_i * (1/z_occluder - 1/z)

    and only that offset MODULO the fence period matters. The circular
    resultant of those phases is 1 when every view sees the same phase - in
    which case the fence survives at full contrast however clever the
    estimator - and 0 when they are spread evenly, which is the best any
    aperture can do. The grid is separable, so x and y are scored apart and
    the worse of the two is returned.

    This is why a plane can be right about the target and still show the
    fence, and why the answer is a different plane, not a different estimator.
    """
    if not np.isfinite(period_px) or period_px <= 1:
        return float("nan")
    k = f_px * (1.0 / occluder_mm - 1.0 / z_mm) * 2 * np.pi / period_px
    xr, yr = cam_xy_mm(ref, pitch_mm)
    ph = [((x - xr) * k, (y - yr) * k)
          for x, y in (cam_xy_mm(i, pitch_mm) for i in cams)]
    rx = abs(np.mean([np.exp(1j * a) for a, _ in ph]))
    ry = abs(np.mean([np.exp(1j * b) for _, b in ph]))
    return float(max(rx, ry))


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


# ------------------------------------------------------------ calibration ---
def load_calibration(path: Path):
    """analyze/calibrate.py's npz -> the intrinsics and extrinsics used here.

    Per camera eNN the file carries camera_matrix, dist_coeffs (OpenCV's
    14-coefficient model: rational radial, tangential, thin prism, tilted
    sensor), image_size and one ChArUco board pose, plus a relative pose
    eII_to_eJJ for every pair of cameras.

    The pair entries are a median over every frame the board was seen in; the
    board poses come from a single representative frame. So the warp leans on
    the pairs, and composing two board poses is only the fallback for a pair
    the file does not carry.

    Translations are metres in the file and millimetres here, to match --z.
    """
    try:
        d = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}")

    def arr(key):
        return np.asarray(d[key], np.float64)

    cams = {}
    for i in range(N_CAM):
        n = cam_name(i)
        if f"{n}_camera_matrix" not in d:
            continue
        size = None
        if f"{n}_image_size" in d:
            wh = np.atleast_1d(d[f"{n}_image_size"]).ravel()
            size = (int(wh[0]), int(wh[1]))
        cams[i] = {
            "K": arr(f"{n}_camera_matrix").reshape(3, 3),
            "dist": arr(f"{n}_dist_coeffs").ravel(),
            "size": size,
            "err": (float(arr(f"{n}_reprojection_error").ravel()[0])
                    if f"{n}_reprojection_error" in d else float("nan")),
        }
    if not cams:
        raise SystemExit(f"{path} holds no eNN_camera_matrix")

    poses = {}
    for i in cams:
        n = cam_name(i)
        if f"{n}_board_rotation_matrix" in d and f"{n}_board_tvec" in d:
            poses[i] = (arr(f"{n}_board_rotation_matrix").reshape(3, 3),
                        arr(f"{n}_board_tvec").ravel() * 1000.0)
    pairs = {}
    for i in range(N_CAM):
        for j in range(N_CAM):
            k = f"{cam_name(i)}_to_{cam_name(j)}"
            if f"{k}_rotation_matrix" in d and f"{k}_tvec" in d:
                pairs[(i, j)] = (arr(f"{k}_rotation_matrix").reshape(3, 3),
                                 arr(f"{k}_tvec").ravel() * 1000.0)
    return {"path": str(path), "cams": cams, "poses": poses, "pairs": pairs}


def relative_pose(calib, cam_i, ref_i):
    """(R, t) with X_i = R X_ref + t, millimetres, or None if unknown."""
    if cam_i not in calib["cams"] or ref_i not in calib["cams"]:
        return None
    if cam_i == ref_i:
        return np.eye(3), np.zeros(3)
    pairs = calib["pairs"]
    if (ref_i, cam_i) in pairs:
        return pairs[(ref_i, cam_i)]
    if (cam_i, ref_i) in pairs:                        # stored the other way
        R, t = pairs[(cam_i, ref_i)]
        return R.T, -(R.T @ t)
    poses = calib["poses"]
    if cam_i in poses and ref_i in poses:              # via the board
        (Ri, ti), (Rr, tr) = poses[cam_i], poses[ref_i]
        R = Ri @ Rr.T
        return R, ti - R @ tr
    return None


def scaled_K(K, size, shape):
    """The intrinsics of the image actually being processed.

    --scale resizes before anything else, and a burst need not come from the
    sensor mode the calibration was shot in; both are the same correction. A
    pixel centre sits at its integer coordinate, so the principal point moves
    as (c + 0.5) * s - 0.5, not c * s. The distortion coefficients live in
    normalised coordinates and do not move at all.
    """
    K = np.asarray(K, np.float64).copy()
    if size is None:
        return K
    h, w = shape
    sx, sy = w / float(size[0]), h / float(size[1])
    K[0, 0] *= sx; K[0, 1] *= sx; K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 1] *= sy; K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def tilt_matrix(tau_x, tau_y):
    """OpenCV's tilted-sensor projection, coefficients 13 and 14.

    calibrate.py fits the tilted model, so these are not zero and dropping
    them leaves a bias of a few pixels out at the corners.
    """
    cx, sx = np.cos(tau_x), np.sin(tau_x)
    cy, sy = np.cos(tau_y), np.sin(tau_y)
    rot_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    rot_y = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    rot = rot_y @ rot_x
    proj = np.array([[rot[2, 2], 0, -rot[0, 2]],
                     [0, rot[2, 2], -rot[1, 2]],
                     [0, 0, 1]])
    return proj @ rot


def distort_normalised(x, y, dist):
    """Normalised camera coordinates -> the distorted ones, OpenCV's model.

    This is the forward direction, which is the cheap one: it is a closed
    form, where undistortion is iterative. Going this way lets the whole warp
    - undistort, rectify, project onto the plane - be a single resample per
    camera instead of two, which for 16 cameras over 90 frames is the
    difference between a run and a wait.
    """
    k = np.zeros(14)
    dist = np.asarray(dist, np.float64).ravel()
    k[:min(14, dist.size)] = dist[:14]
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y = k
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4
    if tau_x or tau_y:
        m = tilt_matrix(tau_x, tau_y)
        w = m[2, 0] * xd + m[2, 1] * yd + m[2, 2]
        w = np.where(w == 0, 1.0, w)                   # OpenCV: 1/z, or 1 if z is 0
        xd, yd = ((m[0, 0] * xd + m[0, 1] * yd + m[0, 2]) / w,
                  (m[1, 0] * xd + m[1, 1] * yd + m[1, 2]) / w)
    return xd, yd


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


def sample_map(calib, cam_i, ref_i, z_mm, shape):
    """Where every pixel of the reference view reads from in camera i.

    A plane at z_mm in front of the reference camera, normal n = (0, 0, 1) in
    that camera's frame, sends the reference ray K_ref^-1 x_ref to the point
    z * K_ref^-1 x_ref, which camera i sees at

        X_i = (R + t n^T / z) K_ref^-1 x_ref

    in its own frame. Normalise, distort, apply K_i, and that is the sensor
    pixel to read. This is the plane-induced homography with the two lens
    models folded in, so the output view is camera i seen through an ideal
    pinhole with K_ref - the reference camera's own undistorted view, since
    for i = ref the same expression reduces to plain undistortion.

    Returns two float32 maps, or None when the file cannot place this camera.
    Anywhere the plane point falls behind camera i or off its sensor the maps
    hold -1, which reads as "outside" in both back ends below.
    """
    rel = relative_pose(calib, cam_i, ref_i)
    if rel is None:
        return None
    R, t = rel
    ref_c, cam_c = calib["cams"][ref_i], calib["cams"][cam_i]
    k_ref = scaled_K(ref_c["K"], ref_c["size"], shape)
    k_cam = scaled_K(cam_c["K"], cam_c["size"], shape)
    m = (R + np.outer(t, (0.0, 0.0, 1.0)) / float(z_mm)) @ np.linalg.inv(k_ref)

    h, w = shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    xc = m[0, 0] * u + m[0, 1] * v + m[0, 2]
    yc = m[1, 0] * u + m[1, 1] * v + m[1, 2]
    zc = m[2, 0] * u + m[2, 1] * v + m[2, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        behind = zc <= 0
        zc = np.where(behind, np.nan, zc)
        xd, yd = distort_normalised(xc / zc, yc / zc, cam_c["dist"])
        mx = k_cam[0, 0] * xd + k_cam[0, 1] * yd + k_cam[0, 2]
        my = k_cam[1, 1] * yd + k_cam[1, 2]
        # A rational radial model is only fitted over the sensor; well outside
        # it the quotient can turn or blow up, so anything that does not land
        # on the image is dropped here rather than folded onto its border.
        bad = ~(np.isfinite(mx) & np.isfinite(my))
        bad |= (mx < -1) | (mx > w) | (my < -1) | (my > h)
    return (np.where(bad, -1.0, mx).astype(np.float32),
            np.where(bad, -1.0, my).astype(np.float32))


_MAP_LOCK = threading.Lock()
_MAP_PLANE = None                                      # (ref, z, shape) held
_MAP_CACHE = {}                                        # camera index -> maps


def cached_sample_map(calib, cam_i, ref_i, z_mm, shape):
    """One plane's maps, built once and reused by every frame of the burst.

    Two float32 planes per camera is ~13 MB at full size, so only the plane in
    hand is kept: a sweep visits each z once and the sequence then sits on the
    winner for all 90 frames. Built under a lock because process_frame warps
    the cameras from a thread pool.
    """
    global _MAP_PLANE
    plane = (ref_i, round(float(z_mm), 6), shape)
    with _MAP_LOCK:
        if _MAP_PLANE != plane:
            _MAP_CACHE.clear()
            _MAP_PLANE = plane
        if cam_i not in _MAP_CACHE:
            _MAP_CACHE[cam_i] = sample_map(calib, cam_i, ref_i, z_mm, shape)
        return _MAP_CACHE[cam_i]


def remap_image(img: np.ndarray, mx: np.ndarray, my: np.ndarray):
    """Sample img at (mx, my), NaN outside - the general form of shift_image."""
    if cv2 is not None:
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    h, w = img.shape
    x0 = np.floor(mx).astype(np.intp)
    y0 = np.floor(my).astype(np.intp)
    inside = (x0 >= 0) & (x0 + 1 < w) & (y0 >= 0) & (y0 + 1 < h)
    xc = np.clip(x0, 0, w - 2)
    yc = np.clip(y0, 0, h - 2)
    fx = (mx - xc).astype(np.float32)
    fy = (my - yc).astype(np.float32)
    a = img[yc, xc]; b = img[yc, xc + 1]
    c = img[yc + 1, xc]; d = img[yc + 1, xc + 1]
    out = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    return np.where(inside, out, np.nan).astype(np.float32)


def warp_for_plane(img, cam_i, ref_i, z_mm, f_px, pitch_mm, calib=None,
                   extra=(0.0, 0.0)):
    """Align camera cam_i onto the reference view for a plane at z_mm.

    Two ways to do it, and the rest of the pipeline does not care which ran:

    ideal    with no --calib the cameras are taken to be coplanar, parallel
             and identical, and the warp is the translation f * B / z that
             follows from the nominal grid.
    calib    with --calib it is the plane-induced homography built from the
             measured intrinsics and relative poses, distortion included.

    The array is not ideal: calibration_results.npz has the cameras 0.4 to 3.3
    degrees off each other, which at f = 1800 px is up to a hundred pixels of
    misalignment across the frame that no translation can take out. The
    translation stays as the fallback for a camera the calibration does not
    cover.

    `extra` is the residual shift --refine measured on the images, applied on
    top of whichever of the two ran. It is kept out of the cached maps so that
    refining, which tries several of them, does not rebuild the maps each time.

    Returns (warped image, (dx, dy)): the translation for the ideal path, and
    for the calibrated one the displacement the map applies at the image
    centre - which also absorbs the two cameras' differing principal points,
    so it is a report of what happened, not the same quantity.
    """
    ex, ey = float(extra[0]), float(extra[1])
    if calib is not None:
        maps = cached_sample_map(calib, cam_i, ref_i, z_mm, img.shape)
        if maps is not None:
            mx, my = maps
            if ex or ey:
                mx, my = mx - ex, my - ey       # read further left = move right
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
            centre = ((float(cx - mx[cy, cx]), float(cy - my[cy, cx]))
                      if mx[cy, cx] >= 0 else (float("nan"), float("nan")))
            return remap_image(img, mx, my), centre
    xi, yi = cam_xy_mm(cam_i, pitch_mm)
    xr, yr = cam_xy_mm(ref_i, pitch_mm)
    dx = f_px * (xi - xr) / z_mm + ex
    dy = f_px * (yi - yr) / z_mm + ey
    return shift_image(img, dx, dy), (dx, dy)


# ------------------------------------------------------------- refinement ---
def match_shift(target: np.ndarray, moving: np.ndarray, radius: int):
    """How far `moving` has to travel to sit on `target`. -> (dx, dy, peak).

    Normalised correlation over a bounded search, with a parabolic fit for the
    sub-pixel part. peak is the correlation at the winner, which is the only
    honest guide to whether the answer means anything.
    """
    t = np.nan_to_num(target).astype(np.float32)
    m = np.nan_to_num(moving).astype(np.float32)
    # target and moving cover the same crop; the template is the target inset
    # by the search radius, so the correlation surface is exactly the shifts
    # in [-radius, +radius] and zero shift sits at its centre.
    tpl = t[radius:t.shape[0] - radius, radius:t.shape[1] - radius]
    if tpl.size == 0:
        return 0.0, 0.0, 0.0
    if cv2 is not None:
        surf = cv2.matchTemplate(m, tpl, cv2.TM_CCOEFF_NORMED)
    else:                                              # same surface, slowly
        surf = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
        a = tpl - tpl.mean()
        an = float(np.sqrt((a * a).sum())) or 1.0
        for r in range(surf.shape[0]):
            for c in range(surf.shape[1]):
                b = m[r:r + tpl.shape[0], c:c + tpl.shape[1]]
                b = b - b.mean()
                surf[r, c] = (a * b).sum() / (an * (np.sqrt((b * b).sum()) or 1.0))
    r0, c0 = np.unravel_index(int(np.argmax(surf)), surf.shape)
    peak = float(surf[r0, c0])

    def sub(axis_vals):
        lo, mid, hi = axis_vals
        d = lo - 2 * mid + hi
        return float(np.clip((lo - hi) / (2 * d), -1, 1)) if d != 0 else 0.0

    sx = sub(surf[r0, c0 - 1:c0 + 2]) if 0 < c0 < surf.shape[1] - 1 else 0.0
    sy = sub(surf[r0 - 1:r0 + 2, c0]) if 0 < r0 < surf.shape[0] - 1 else 0.0
    return radius - (c0 + sx), radius - (r0 + sy), peak


def refine_alignment(images: dict, cams_of_band: dict, ref: int, z_mm: float,
                     f_px: float, pitch_mm: float, calib, ksize: int = 31,
                     radius: int = 24, min_peak: float = 0.2, passes: int = 2):
    """Per-camera residual shift, measured on the burst itself.

    calibration_results.npz was shot on a different prototype. Its intrinsics
    and the rough shape of the array are worth having, but its poses are not
    this array's poses, and what a pose error leaves behind after the plane
    warp is, to first order, a constant shift per camera - which is a thing
    the images themselves can measure.

    Every view is warped with the calibration as it stands, locally normalised
    (the only form comparable across bands, and it also removes the per-band
    gain), and matched against the consensus of the others. Two passes: the
    first against the reference camera alone, the second against the mean of
    the already-corrected views, which is far less noisy.

    The search is bounded on purpose. A fence is periodic, so a wide search
    locks onto the neighbouring bar as happily as onto the right one; a
    residual larger than half the fence period is not something correlation
    can resolve, and pretending otherwise produces a confident wrong answer.
    A camera whose peak is below min_peak is left where the calibration put
    it, and every peak is reported so that call is visible.

    ksize here is deliberately not the compositing --ksize. The local
    normalisation removes everything larger than its window, so a window
    smaller than the occluder's own period throws the occluder away and leaves
    the match nothing but the fine detail - which in a scene lit by specular
    glints is the one thing that is NOT shared between viewpoints. Keep it at
    or above the period of whatever structure the views have in common.

    Returns {camera index: (dx, dy, peak, why)}, where why is empty when the
    shift was applied and says what disqualified it otherwise.
    """
    shifts = {i: (0.0, 0.0, 0.0, "") for i in images}
    h, w = images[ref].shape
    # the middle of the frame, inset by the search radius and by enough border
    # that the crop still has structure to match on
    ch, cw = int(h * 0.62), int(w * 0.62)
    y0, x0 = (h - ch) // 2, (w - cw) // 2

    def prepared(i, extra):
        img = warp_for_plane(images[i], i, ref, z_mm, f_px, pitch_mm, calib,
                             extra)[0]
        return np.nan_to_num(local_normalise(img, ksize), nan=0.0)

    target = None
    for p in range(passes):
        views = {i: prepared(i, shifts[i][:2]) for i in images}
        if target is None:
            target = views[ref]
        total = sum(views.values())
        for i in sorted(images):
            if i == ref:
                continue
            # the consensus of the others, without summing 15 images per camera
            tgt = target if p == 0 else (total - views[i]) / (len(views) - 1)
            dx, dy, peak = match_shift(tgt[y0:y0 + ch, x0:x0 + cw],
                                       views[i][y0:y0 + ch, x0:x0 + cw], radius)
            # A winner sitting on the edge of the search is not a measurement:
            # the surface is still climbing when it runs out of room, so the
            # true optimum is somewhere outside and this number is only where
            # the window happened to stop.
            why = ""
            if max(abs(dx), abs(dy)) >= radius - 0.5:
                why = "search boundary"
            elif peak < min_peak:
                why = "peak below threshold"
            sx, sy = shifts[i][:2]
            if why:
                shifts[i] = (sx, sy, peak, why)
            else:
                shifts[i] = (sx + dx, sy + dy, peak, "")
    return shifts


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


def _view_mean(stack: np.ndarray):
    """Mean across the view axis over the finite entries, NaN where there are

    none. numpy's nanmean warns on an all-NaN column, and after a warp at a
    near plane a wide border has exactly that - every frame would print the
    same warning four times.
    """
    ok = np.isfinite(stack)
    cnt = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.where(ok, stack, 0.0).sum(axis=0) / cnt, np.nan)


def _nan_box(img: np.ndarray, k: int):
    """Window mean over the valid samples only, and how many there were.

    The warp leaves NaN wherever a view has no data; averaging over the window
    instead of over the valid samples would let one missing pixel poison a
    whole k x k neighbourhood.
    """
    valid = np.isfinite(img).astype(np.float64)
    filled = np.where(valid > 0, img, 0.0).astype(np.float64)
    n = _box_sum(valid, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = _box_sum(filled, k) / n
    mean = mean.astype(np.float32)
    mean[n < 0.5] = np.nan
    return mean, n


def local_composite(stack: np.ndarray, ksize: int):
    """Combine one band's views by local agreement, not by a per-pixel statistic.

    A per-pixel median discards what makes an occluder an occluder: it is
    spatially coherent. A fence bar covers a whole neighbourhood of a view,
    not one pixel, so whether a view can be trusted here is a question about
    a window, not about a sample. Each view is scored by the local energy of
    its disagreement with the consensus, and the composite is the weighted
    mean of the views that agree over that window. The views that do see the
    background then contribute in full, instead of the answer being whichever
    single sample happened to sort into the middle of four.

    The weight is scale free, which is the point:

        w_i = exp(-(e_i - e_min) / (e_min + f))

    e_min is the best-agreeing view at that pixel, so a view is judged against
    what agreement is achievable there rather than against an absolute
    threshold. Without that, a bright specular region - where every view
    disagrees by a lot in raw units - would be declared occluded everywhere,
    and a dark region would be declared clean however badly it disagreed. f is
    the median of e_min over the image and stops the quotient from amplifying
    noise where there is no signal at all.

    Returns (composite, per-view weights).
    """
    med, _, cnt = robust_stats(stack)
    energy = np.stack([_nan_box((v - med) ** 2, ksize)[0] for v in stack])
    with np.errstate(invalid="ignore"):
        e_min = np.nanmin(np.where(np.isfinite(energy), energy, np.inf), axis=0)
    e_min = np.where(np.isfinite(e_min), e_min, 0.0)
    finite = np.isfinite(e_min) & (e_min > 0)
    floor = float(np.median(e_min[finite])) if finite.any() else 0.0
    floor = max(floor, 1e-12)
    with np.errstate(invalid="ignore", over="ignore"):
        w = np.exp(-(energy - e_min) / (e_min + floor))
    w = np.where(np.isfinite(w) & np.isfinite(stack), w, 0.0).astype(np.float32)
    tot = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        comp = np.where(tot > 0, (w * np.nan_to_num(stack)).sum(axis=0) / tot, np.nan)
    return np.where(np.isfinite(comp), comp, med).astype(np.float32), w


def occlusion_fraction(stack: np.ndarray, comp: np.ndarray, ksize: int):
    """The share of the light this band's views lose to something in front.

    The old within-band IQR was neither signed nor relative, and so ranked a
    bright specular glint above a dark fence bar - it showed the highlights
    and hid the occluder, which is backwards. An occluder is specifically a
    thing that takes light away, over an area, so this counts only the darker
    side, over a window, and divides by the local light level. A bar covering
    k of the band's V views then reads about k/V whatever the scene
    brightness, and 0 means every view saw the same thing.
    """
    lost = np.stack([_nan_box(np.maximum(comp - v, 0.0), ksize)[0] for v in stack])
    level = _nan_box(comp, ksize)[0]
    finite = np.isfinite(comp)
    # A soft floor tied to the band's own brightness: without it the quotient
    # runs away in the black border, where both terms are noise.
    ref = float(np.percentile(comp[finite], 90)) if finite.any() else 1.0
    floor = max(0.05 * abs(ref), 1e-6)
    with np.errstate(invalid="ignore"):
        occ = _view_mean(lost) / np.sqrt(level * level + floor * floor)
    return np.clip(occ, 0.0, 1.0).astype(np.float32)


def bright_composite(stack: np.ndarray, ksize: int, base_ksize: int = 0):
    """Compose from the upper tail, because occlusion only ever removes light.

    A median assumes the contamination is two-sided and that fewer than half
    the samples are bad. Neither holds here. The fence sits at a tenth of the
    target's distance and is defocused to about its own period, so it is not
    an opaque bar that some views miss: it is a TRANSMISSION field, roughly
    half-open, different in every view because 40 mm of baseline slides it by
    hundreds of pixels. Contamination is therefore multiplicative, one-sided,
    and present in every view at once - so the answer lies in the upper half
    of the views, and a median that treats the middle as the truth is looking
    in the wrong place.

    The brightest view is not quite it either, because the target is a glossy
    ellipsoid: its highlight sits somewhere different on it for every
    viewpoint, so the single brightest sample is as likely to be a specular
    flash as a clear view of a capsule. Measured on burst0001 at 550 mm the
    upward disagreement between views runs about twice the downward one -
    gloss outweighs occlusion in this scene - and keeping only the brightest
    view returns a picture of highlights with the capsule bodies lost between
    them.

    So the extremes at BOTH ends are dropped, the top one for gloss and the
    bottom one for the heaviest occlusion, and what is left is averaged. With
    four views per band that is the second and third brightest.

    The ranking is on each view's local window level RELATIVE TO ITS OWN
    broad level over base_ksize, and both halves of that matter.

    Local, because the choice has to be spatially coherent: a view that is
    occluded here is occluded over a neighbourhood, and switching views pixel
    by pixel on noise would shred the target instead of showing it.

    Relative, because a bare local level ranks the CAMERAS, not the scene. The
    per-camera gain match is one scalar, so any difference in vignetting, in
    illumination falloff or in how much defocused fence a given view is
    looking through survives it - and on burst0001 that was enough to have
    e10 ranked brightest over 43% of the frame and e02 over 9%, when an
    occlusion-driven choice would sit near 25% each. Dividing by the view's
    own broad level takes all of that out at once, including the defocused
    fence's own smooth attenuation, and the split comes back to 29/27/22/22.
    It doubles the detail in the result, because the estimator stops spending
    its rejection on whichever camera happens to be brightest.

    The pixels themselves
    are then taken as they are, which is the other half of why this beats a
    median: a median of four imperfectly aligned views blends them, and the
    capsule edges go with it, while selecting whole views keeps whatever
    sharpness the alignment left.
    """
    v = len(stack)
    base_ksize = base_ksize or (8 * ksize + 1)
    level = np.stack([_nan_box(x, ksize)[0] for x in stack])
    broad = np.stack([_nan_box(x, base_ksize)[0] for x in stack])
    with np.errstate(invalid="ignore", divide="ignore"):
        level = level / np.where(np.abs(broad) > 1e-6, broad, np.nan)
    level = np.where(np.isfinite(level), level, -np.inf)
    order = np.argsort(level, axis=0)                  # ascending, dark first
    drop_top = 1 if v >= 4 else 0
    drop_bottom = 1 if v >= 3 else 0
    ranks = list(range(drop_bottom, max(drop_bottom + 1, v - drop_top)))
    picked = [np.take_along_axis(stack, order[r:r + 1], axis=0)[0] for r in ranks]
    picked = np.stack(picked)
    good = np.isfinite(picked)
    tot = good.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(tot > 0, np.where(good, picked, 0.0).sum(axis=0) / tot, np.nan)
    return out.astype(np.float32)


def gloss_fraction(stack: np.ndarray, comp: np.ndarray, ksize: int):
    """The bright counterpart of occlusion_fraction: view-dependent specular.

    A glossy ellipsoid carries a highlight that sits at a different place on
    it for every viewpoint, so a capsule shows up as light that ONE view has
    and the others do not - the exact mirror of an occluder, which is light
    the others have and one view does not. Separating the two sides of the
    same disagreement turns a nuisance into a map of where the glossy things
    are, and keeps the highlight out of the occlusion layer where it used to
    masquerade as a fence.
    """
    gain = np.stack([_nan_box(np.maximum(v - comp, 0.0), ksize)[0] for v in stack])
    level = _nan_box(comp, ksize)[0]
    finite = np.isfinite(comp)
    ref = float(np.percentile(comp[finite], 90)) if finite.any() else 1.0
    floor = max(0.05 * abs(ref), 1e-6)
    with np.errstate(invalid="ignore"):
        g = _view_mean(gain) / np.sqrt(level * level + floor * floor)
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def notch_fence(img: np.ndarray, period_px: float, harmonics: int = 3,
                width: float = 0.22):
    """Remove the occluder's periodic component from a composite.

    What the aperture cannot average away, symmetry can. A woven fence is a
    near-regular lattice: in the Fourier plane it is a handful of sharp peaks
    on the two lattice axes and their harmonics, while the target - twenty
    millimetres of glossy capsule, in no particular arrangement - is
    broadband. Zeroing those peaks takes out the fence and costs the target
    only the sliver of its spectrum that happens to coincide.

    This is the one tool that separates an occluder sitting at the SAME depth
    as the target, where no choice of plane can help, and it is opt-in for
    exactly that reason: it will also erase a genuinely periodic subject.

    The notches are Gaussian, on the two axes and their mirror images, with
    the DC row and column left alone so the illumination is not touched.
    """
    a = np.nan_to_num(img).astype(np.float32)
    h, w = a.shape
    F = np.fft.fftshift(np.fft.fft2(a))
    cy, cx = h // 2, w // 2
    fy, fx = np.ogrid[:h, :w]
    mask = np.ones((h, w), np.float32)
    sigma = max(width * min(h, w) / period_px, 1.0)
    for n in range(1, harmonics + 1):
        step_x = n * w / period_px
        step_y = n * h / period_px
        for dx, dy in ((step_x, 0.0), (-step_x, 0.0), (0.0, step_y), (0.0, -step_y)):
            if abs(dx) > w / 2 or abs(dy) > h / 2:
                continue
            mask *= 1.0 - np.exp(-(((fx - cx - dx) ** 2 + (fy - cy - dy) ** 2)
                                   / (2 * sigma * sigma)))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask))).astype(np.float32)
    return np.where(np.isfinite(img), out, np.nan)


def find_capsules(img: np.ndarray, size_px: float, gloss: np.ndarray = None,
                  size_tol=(0.55, 1.9), max_ratio=3.0, gloss_max=0.55,
                  min_snr=1.6, level_band=(90.0, 99.0)):
    """Find target-sized elliptical blobs, and say which ones are highlights.

    The scene says how big a capsule is - 20 mm at the focus plane, so tens of
    pixels - and that is nearly all of the detector. A difference of Gaussians
    tuned to that size answers "is there a blob of about this width here",
    which suppresses the fence (finer, and periodic) and the illumination
    (broader) in one step, without either being modelled.

    The trap is the specular highlight. A capsule is glossy, so the brightest
    thing in the frame is often not a capsule but the flash off one - and
    worse, the composite's own rejection of that flash leaves a rounded bright
    patch behind, because the view carrying the highlight is dropped over the
    whole window and the ones that replace it were never lit the same way.
    Those patches are round, bright and about the size of the rejection
    window, so a brightness-and-roundness detector will happily call them
    capsules.

    Brightness is a third filter, and a blunt one on purpose. The top of the
    range is clipped - on burst0500 about 0.4% of every frame sits exactly at
    the sensor ceiling - and a clipped blob has no shape left to measure: its
    edges are where the clipping stopped, not where the object did. The bottom
    of the range is not an object at all, it is the read noise of a scene whose
    median sits at fourteen counts. So only blobs whose level falls between two
    percentiles of the frame are measured, and the rest are reported with the
    reason rather than silently dropped.

    Two more things separate a capsule from a fence cell, and both are used:

    size     a highlight is a few millimetres of a twenty millimetre capsule,
             and the hole-fill patch is the size of the compositing window.
             Both sit well below the passband a capsule occupies.
    gloss    a real capsule body is view-CONSISTENT - it is at the focus
             plane, that is what focusing means - and carries a small
             view-dependent highlight somewhere inside it. An artefact is
             view-dependent over its whole area. So a blob whose area is
             mostly high-gloss is reported as a highlight, not a capsule.

    Returns a list of dicts with centre, axes, angle, and why each was kept or
    not. Nothing here is a segmentation: it is a reading aid, and a blob it
    marks is a claim about scale and consistency, not about identity.
    """
    a = np.nan_to_num(img).astype(np.float32)
    s1, s2 = size_px / 4.0, size_px / 1.4
    dog = cv2.GaussianBlur(a, (0, 0), s1) - cv2.GaussianBlur(a, (0, 0), s2) \
        if cv2 is not None else None
    if dog is None:
        return []
    med = float(np.median(dog))
    mad = float(np.median(np.abs(dog - med))) * 1.4826
    thr = med + min_snr * max(mad, 1e-6)
    mask = (dog > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)

    lo, hi = size_tol[0] * size_px, size_tol[1] * size_px
    area_lo, area_hi = 0.25 * np.pi / 4 * lo * lo, np.pi / 4 * hi * hi
    gl = None if gloss is None else np.nan_to_num(gloss)
    gl_thr = float(np.median(gl)) if gl is not None else 0.0
    finite = np.isfinite(img)
    band_lo, band_hi = np.percentile(a[finite], level_band) if finite.any() else (0, 1)
    ceiling = float(a[finite].max()) if finite.any() else float("inf")
    out = []
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    for k in range(1, n):
        area = float(stats[k, cv2.CC_STAT_AREA])
        if area < area_lo or area > area_hi:
            continue
        m = lab == k
        y, x = ys[m].astype(np.float64), xs[m].astype(np.float64)
        cy, cx = y.mean(), x.mean()
        # second moments -> the ellipse with the same inertia as the blob
        cyy, cxx = ((y - cy) ** 2).mean(), ((x - cx) ** 2).mean()
        cxy = ((x - cx) * (y - cy)).mean()
        tr, det = cyy + cxx, cyy * cxx - cxy * cxy
        disc = max(tr * tr / 4 - det, 0.0) ** 0.5
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-9)
        major, minor = 4 * l1 ** 0.5, 4 * l2 ** 0.5      # 2 sigma each way
        ratio = major / max(minor, 1e-9)
        ang = float(np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy)))
        vals = a[m]
        mean_level, peak = float(vals.mean()), float(vals.max())
        why = ""
        if not (lo <= major <= hi):
            why = "wrong size"
        elif ratio > max_ratio:
            why = "too elongated"
        elif peak >= ceiling:
            why = "clipped: no shape left to measure"
        elif mean_level > band_hi:
            why = "brighter than the usable band"
        elif mean_level < band_lo:
            why = "dimmer than the usable band"
        elif gl is not None and float(np.mean(gl[m] > gl_thr)) > gloss_max:
            why = "view-dependent: a highlight, not a body"
        out.append({"x": float(cx), "y": float(cy), "major": float(major),
                    "minor": float(minor), "angle": ang, "area": area,
                    "level": mean_level, "peak": peak,
                    "gloss": float(np.mean(gl[m])) if gl is not None else float("nan"),
                    "why": why})
    return out


def persistence_filter(found, flanks, tol_px):
    """Keep only the blobs that stay put when the focus plane is moved.

    This is the one test that actually asks about depth, and it is needed
    because on this rig scale cannot answer the question: the fence's cells
    are 3.5 mm at 100 mm, which projects to 60 px, and a 20 mm capsule at
    550 mm projects to 65 px. A size-and-roundness detector cannot tell them
    apart, and measured on burst0001 it does not - 42% of what it kept sat on
    the fence lattice against 9% by chance.

    What does tell them apart is what happens when the plane moves. A target
    at the focus plane is at the focus plane: shifting the plane by a few tens
    of millimetres blurs it slightly and leaves its centroid where it was. The
    fence is at a fifth of that distance, so the same shift slides it by tens
    of pixels in every view and the aliased pattern it leaves in the composite
    reorganises completely.

    So a blob is reported as being at this depth only if it is also found, in
    the same place, in the composites either side of the plane. A blob that
    fails this is not noise - it is a real feature of the composite - but it
    is not at the depth the plane says, and calling it a capsule would be a
    guess dressed up as a measurement.
    """
    for f in found:
        if f["why"]:
            continue
        held = 0
        for other in flanks:
            near = [g for g in other
                    if (g["x"] - f["x"]) ** 2 + (g["y"] - f["y"]) ** 2 <= tol_px ** 2]
            held += bool(near)
        f["held"] = held
        if held < len(flanks):
            f["why"] = f"not at this depth: held {held} of {len(flanks)} planes"
    return found


def draw_capsules(img_u8: np.ndarray, found, show_rejected=True):
    """The detections over the image they came from, as a reading aid."""
    out = img_u8 if img_u8.ndim == 3 else np.dstack([img_u8] * 3)
    out = np.ascontiguousarray(out)
    if cv2 is None:
        return out
    kept = 0
    for f in found:
        ok = not f["why"]
        if not ok and not show_rejected:
            continue
        kept += ok
        # RGB, not BGR: write_png is the one that reverses, for cv2's benefit.
        if ok:
            colour = (110, 255, 110)                     # green: held its ground
        elif f["why"].startswith("view-dependent"):
            colour = (120, 190, 255)                     # blue: a highlight
        elif f["why"].startswith("not at this depth"):
            colour = (255, 110, 110)                     # red: wrong depth
        elif f["why"].startswith(("clipped", "brighter", "dimmer")):
            colour = (230, 230, 120)                     # yellow: unusable level
        else:
            colour = (160, 160, 160)                     # grey: wrong shape
        cv2.ellipse(out, ((f["x"], f["y"]), (f["minor"], f["major"]),
                          f["angle"] + 90.0), colour, 1, cv2.LINE_AA)
    depth = sum(1 for f in found if f["why"].startswith("not at this depth"))
    spec = sum(1 for f in found if f["why"].startswith("view-dependent"))
    txt = (f"green {kept} at this depth | red {depth} target-sized but not at "
           f"this depth | blue {spec} specular")
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(out, txt, (12, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, col, th, cv2.LINE_AA)
    return out


def tone_map(u8: np.ndarray, gamma: float = 1.0, clahe: float = 0.0,
             tile: int = 8):
    """Display shaping, applied after the range map and never before it.

    This scene is dark - the composite's median sits around ten counts out of
    a thousand - and a linear map spends almost all of its 256 levels on a
    range the scene never uses. Gamma is a fixed monotone curve, so it lifts
    the shadows without breaking --stretch fixed: the same input still maps to
    the same output in every frame of a sequence, and the video does not
    flicker.

    CLAHE does not have that property. It equalises against a neighbourhood,
    so its mapping changes with the content and a sequence built on it will
    breathe. It is worth it on a single frame and a bad idea on a video.

    The tile grid matters more than the clip limit: tiles smaller than the
    target equalise the target itself away, which is why the caller sets the
    grid from the scene - a few target widths per tile - rather than leaving
    OpenCV's default.
    """
    out = u8
    if gamma and gamma != 1.0:
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0,
                      0, 255).astype(np.uint8)
        out = lut[out]
    if clahe > 0 and cv2 is not None:
        c = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(tile, tile))
        if out.ndim == 3:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = c.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            out = c.apply(out)
    return out


def band_composites(warped: dict, cams_of_band: dict, ksize: int = 15,
                    combine: str = "local", base_ksize: int = 0):
    """Per band: the composite, the occlusion map and the gloss map.

    All three are photometric statements, so all three stay inside one band.
    "spread" is the occlusion fraction, not a raw IQR, and "gloss" is its
    bright-side twin - the two halves of the same disagreement, which for this
    scene are two different physical things: a fence taking light away, and a
    capsule's highlight adding it.
    """
    comp, spread, gloss, nviews = {}, {}, {}, {}
    for band, idxs in cams_of_band.items():
        have = [i for i in idxs if i in warped]
        if not have:
            continue
        stack = np.stack([warped[i] for i in have])
        if combine == "median" or len(have) < 2:
            med, _, _ = robust_stats(stack)
        elif combine == "bright":
            med = bright_composite(stack, ksize, base_ksize)
        else:
            med, _ = local_composite(stack, ksize)
        comp[band] = med
        spread[band] = occlusion_fraction(stack, med, ksize)
        gloss[band] = gloss_fraction(stack, med, ksize)
        nviews[band] = len(have)
    return comp, spread, gloss, nviews


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


def focus_cost(warped: dict, cams_of_band: dict, ksize: int = 15,
               notch_period=None):
    """How well the plane explains the scene, judged WITHIN bands only.

    Not the median within-band spread, which is the wrong question in an
    occluded scene: the fence disagrees at every plane, so a spread averaged
    over all views is dominated by the occluder and by the specular highlights
    rather than by whether the background lined up. The question that matters
    is whether the views that CAN see the background agree there. So per pixel
    this takes the smallest local disagreement across the band's views - the
    best-agreeing view against the consensus - and divides it by the local
    light level, which stops a bright patch from outvoting the whole frame.

    Two things this has to be told, or it answers a different question:

    notch_period  the fence is the loudest thing in the frame at every plane,
                  so an untouched cost scores how well the FENCE lined up, not
                  the target. Notching the fence out of each view first leaves
                  the cost to judge what is actually being focused on.
    the border    a near plane throws the views far apart, so a wide margin has
                  fewer of them contributing; scoring there would reward near
                  planes for having less to disagree about. Only pixels seen by
                  every view of the band count.

    Cross-band variance would be dominated by the filters and would minimise
    at the wrong depth, so this still never leaves a band.
    """
    per_band = []
    for band, idxs in cams_of_band.items():
        have = [i for i in idxs if i in warped]
        if len(have) < 2:
            continue
        stack = np.stack([warped[i] for i in have])
        full = np.isfinite(stack).all(axis=0)
        if not full.any():
            continue
        if notch_period:
            stack = np.stack([notch_fence(v, notch_period) for v in stack])
        med, _, _ = robust_stats(stack)
        energy = np.stack([_nan_box((v - med) ** 2, ksize)[0] for v in stack])
        with np.errstate(invalid="ignore"):
            e_min = np.nanmin(np.where(np.isfinite(energy), energy, np.inf), axis=0)
            level = _nan_box(med, ksize)[0]
        finite = np.isfinite(med)
        ref = float(np.percentile(med[finite], 90)) if finite.any() else 1.0
        floor = max(0.05 * abs(ref), 1e-6)
        with np.errstate(invalid="ignore"):
            rel = np.sqrt(np.where(np.isfinite(e_min), e_min, np.nan)) / \
                np.sqrt(level * level + floor * floor)
            good = rel[full & np.isfinite(rel)]
            if good.size:
                per_band.append(float(np.median(good)))
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


def sweep_planes(spec: str, scene: dict, f_px: float, pitch_mm: float,
                 scale: float, grid=GRID):
    """The planes to try: "auto", "start:stop", or the explicit "a:b:step".

    A step of so many millimetres samples the wrong thing. What a plane change
    does to the images is a change of DISPARITY, f*span/z, and that is not
    linear in z: over 400-700 mm a 100 mm step moves the aperture's extremes
    by 30-45 px, while over 3000-3100 mm the same step moves them by 3 px. A
    sweep uniform in z therefore spends nearly all its samples where nothing
    changes and skips over the range where everything does.

    So the planes are laid out uniformly in 1/z, with the step set to a
    quarter of a target width of disparity across the aperture - fine enough
    that the target cannot slip between two planes.
    """
    parts = [q for q in spec.split(":") if q] if spec != "auto" else []
    if len(parts) == 3:
        return parse_sweep(spec)
    lo = float(parts[0]) if len(parts) >= 2 else scene["sweep_near_mm"]
    hi = float(parts[1]) if len(parts) >= 2 else scene["sweep_far_mm"]
    span = pitch_mm * (grid - 1)
    d_step = 0.25 * scene["target_px"] * scale / (f_px * span)   # in 1/mm
    a, b = 1.0 / max(hi, 1e-6), 1.0 / max(lo, 1e-6)
    n = max(2, int(round((b - a) / max(d_step, 1e-12))) + 1)
    n = min(n, 400)                                    # a sweep, not a career
    return [1.0 / v for v in np.linspace(b, a, n)]


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


def load_offsets(path: Path, cams):
    """analyze/temporal_align.py's json -> {camera index: frame offset}.

    The Pis answer the same trigger but start their bursts a little apart, so
    frame k is not the same instant on every camera. temporal_align.py
    measures that as an integer offset per camera from the SCENE's own motion
    - the tumbler's - which is the one thing every viewpoint and every band
    agrees on. On burst0001 the offsets run -2 to +2, and at 33 ms a frame
    that is 66 ms of a rotating drum, which no amount of spatial alignment
    can undo.
    """
    try:
        d = json.loads(Path(path).read_text())
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    raw = d.get("offsets", d)
    off = {}
    for i in cams:
        v = raw.get(cam_name(i))
        if v is not None:
            off[i] = int(v)
    if not off:
        raise SystemExit(f"{path} names none of the cameras in this burst")
    return off, d


def load_all_cameras(cams, idx, scale, pool, offsets=None):
    """One frame from every camera, in parallel - numpy releases the GIL.

    offsets shifts each camera's frame index so that one call returns the same
    INSTANT everywhere rather than the same index.
    """
    offsets = offsets or {}

    def one(i):
        path, prefix = cams[i]
        img, n, meta = load_camera_frame(path, prefix, idx + offsets.get(i, 0))
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
                  ksize, calib, pool, extra=None, combine="local",
                  notch_period=None, base_ksize=0, detect_px=0.0,
                  detect_planes=(), offsets=None):
    """One frame -> {layer name: image}. The whole per-frame pipeline."""
    images, metas, _ = load_all_cameras(cams, idx, scale, pool, offsets)
    images, gains = band_gain_normalise(images, cams_of_band)
    extra = extra or {}

    def warp(i):
        return i, warp_for_plane(images[i], i, ref, z, f_px, pitch, calib,
                                 extra.get(i, (0.0, 0.0)))

    warped, shifts = {}, {}
    for i, (w, sh) in [f.result() for f in [pool.submit(warp, i) for i in sorted(images)]]:
        warped[i], shifts[i] = w, sh

    comp, spread, gloss, _ = band_composites(warped, cams_of_band, ksize, combine,
                                            base_ksize)
    layers = {}
    for band, img in comp.items():
        layers[f"band{band}"] = img
    for band, img in spread.items():
        layers[f"spread{band}"] = img
    for band, img in gloss.items():
        layers[f"gloss{band}"] = img
    cross = cross_band_composite(comp, ksize)
    if cross is not None:
        layers["crossband"] = cross
    if notch_period:
        # The fence that survives the aperture is the one sitting at the
        # target's own depth; symmetry is the only handle left on it.
        for band, img in comp.items():
            layers[f"defence{band}"] = notch_fence(img, notch_period)
        if cross is not None:
            layers["defence_crossband"] = notch_fence(cross, notch_period)
    found = []
    if detect_px and comp:
        # Detect on the brightest band that is present, with the fence out of
        # the way if it has been taken out - a lattice of the wrong scale
        # still leaks into a band-pass, and the notch is free here.
        band = max(comp, key=lambda b: float(np.nanpercentile(comp[b], 90)))
        base = layers.get(f"defence{band}", comp[band])
        found = find_capsules(base, detect_px, gloss.get(band))
        flanks = []
        for zz in detect_planes:
            fw = {i: warp_for_plane(images[i], i, ref, zz, f_px, pitch, calib,
                                    extra.get(i, (0.0, 0.0)))[0]
                  for i in cams_of_band[band] if i in images}
            fc, _, fg, _ = band_composites(fw, {band: cams_of_band[band]}, ksize,
                                           combine, base_ksize)
            fb = notch_fence(fc[band], notch_period) if notch_period else fc[band]
            flanks.append(find_capsules(fb, detect_px, fg.get(band)))
        if flanks:
            found = persistence_filter(found, flanks, max(6.0, 0.15 * detect_px))
        layers["capsules"] = (base, found)              # drawn at write time
    return layers, comp, gains, shifts, found


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("burst", type=Path,
                    help="packed burst directory, or a merged burstNNNN.npz")
    ap.add_argument("--z", type=float, default=3000.0,
                    help="focal plane distance in mm (default 3000)")
    ap.add_argument("--sweep", default=None, nargs="?", const="auto",
                    help="pick the plane first. start:stop:step in mm, or start:stop and "
                         "the step is chosen so consecutive planes differ by a quarter "
                         "of a target width of disparity, or bare --sweep for the range "
                         "the stated setup implies")
    ap.add_argument("--save-sweep", action="store_true",
                    help="write the composite at every swept plane, not just the winner")
    ap.add_argument("--occluder-mm", type=float, default=100.0,
                    help="lens front to the tumbler fence, mm (default 100)")
    ap.add_argument("--target-mm", type=float, default=550.0,
                    help="lens front to the target, mm (default 550)")
    ap.add_argument("--target-size-mm", type=float, default=20.0,
                    help="how big the target is, mm (default 20, a soft capsule). "
                         "Sets every local window and the sweep step")
    ap.add_argument("--keep-fence", action="store_true",
                    help="score the sweep on the images as they are. By default the "
                         "fence is notched out of each view first, because it is the "
                         "loudest thing in the frame and otherwise the sweep measures "
                         "how well the FENCE lined up, not the target")
    ap.add_argument("--offsets", default=None, nargs="?", const="auto",
                    help="apply analyze/temporal_align.py's per-camera frame offsets, "
                         "so one frame index is one INSTANT rather than one index. "
                         "Give the json, or bare --offsets to look for "
                         "sa_results/<pack id>/temporal_align_b<NNNN>.json")
    ap.add_argument("--detect", action="store_true",
                    help="find target-sized elliptical blobs and draw them over the "
                         "composite as the 'capsules' layer. Each is checked against "
                         "two flanking planes, because on this rig a fence cell and a "
                         "capsule are the same size in pixels and only depth tells "
                         "them apart. Costs three composites per frame instead of one")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="display gamma, applied after the range map (default 1 = "
                         "linear). 2 to 2.5 opens up a scene this dark. A fixed curve, "
                         "so it does not make a sequence flicker")
    ap.add_argument("--clahe", type=float, default=0.0,
                    help="local contrast equalisation, clip limit (2 is a good start, "
                         "0 = off); the tile grid comes from the target size. Adapts "
                         "to each frame, so it will breathe in a video - for stills. "
                         "--gamma 1.6 --clahe 2 reads best on this scene")
    ap.add_argument("--notch-fence", action="store_true",
                    help="also write defence<nm>: the composite with the fence's own "
                         "lattice frequencies notched out. The only handle on an "
                         "occluder that sits at the target's depth")
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
                    help="window for the local normalisation, the local composite "
                         "and the occlusion map (default 15)")
    ap.add_argument("--combine", choices=("bright", "local", "median"),
                    default="bright",
                    help="how a band's views become one image. 'bright' (default) ranks "
                         "the views by their local level and averages the middle of the "
                         "upper half - right when the occluder attenuates rather than "
                         "replaces, and the target is glossy. 'local' weighs every view "
                         "by how well it agrees over a --ksize window. 'median' is the "
                         "old per-pixel median")
    ap.add_argument("--refine", action="store_true",
                    help="measure a residual per-camera shift on the burst itself and "
                         "apply it - the calibration is from another prototype, and "
                         "what its pose error leaves is a shift per camera")
    ap.add_argument("--refine-ksize", type=int, default=31,
                    help="local-normalisation window for the matching (default 31). "
                         "Smaller than the occluder's period and the match sees only "
                         "the view-dependent fine detail")
    ap.add_argument("--refine-radius", type=int, default=24,
                    help="search radius in px for --refine (default 24; keep it "
                         "under half the fence period or it locks onto the next bar)")
    ap.add_argument("--refine-peak", type=float, default=0.2,
                    help="correlation a camera must reach before --refine moves it "
                         "(default 0.2)")
    ap.add_argument("--refine-from", type=Path, default=None,
                    help="reuse the refine.json of an earlier run instead of "
                         "measuring again")
    ap.add_argument("--stretch", choices=("fixed", "per-frame"), default="fixed",
                    help="8-bit mapping: fixed over the sequence (default, no flicker) "
                         "or recomputed per frame")
    ap.add_argument("--workers", type=int, default=0,
                    help="threads for the per-camera work (default: cameras present)")
    ap.add_argument("--calib", type=Path, default=None,
                    help="calibration_results.npz from analyze/calibrate.py: warp by the "
                         "measured intrinsics and poses instead of the nominal grid")
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

    calib = load_calibration(args.calib) if args.calib else None

    offsets = {}
    if args.offsets:
        pack = args.burst.parent.name
        digits = "".join(c for c in label if c.isdigit())
        opath = (Path("sa_results") / pack / f"temporal_align_b{int(digits or 0):04d}.json"
                 if args.offsets == "auto" else Path(args.offsets))
        if args.offsets == "auto" and not opath.exists():
            raise SystemExit(f"no {opath} -- run analyze/temporal_align.py first, "
                             f"or give --offsets PATH")
        offsets, ometa = load_offsets(opath, cams)
        spread = sorted(offsets.values())
        print(f"offsets : {opath}  reference {ometa.get('ref', '?')}, "
              f"residual RMS {ometa.get('residual_rms', float('nan')):.2f} frame(s), "
              f"range {spread[0]:+d}..{spread[-1]:+d}")
        moved = "  ".join(f"{cam_name(i)}{offsets[i]:+d}"
                          for i in sorted(offsets) if offsets[i])
        print(f"          {moved or 'every camera is already on the same frame'}")

    workers = args.workers or max(1, len(cams))
    pool = ThreadPoolExecutor(max_workers=workers)

    # how many frames the burst holds, and which of them to do
    probe_path, probe_prefix = cams[sorted(cams)[0]]
    _, n_frames, _ = load_camera_frame(probe_path, probe_prefix, 0)
    spec = args.frames if args.frames is not None else (
        str(args.frame) if args.frame is not None else None)
    frames = parse_frames(spec, n_frames)
    if offsets:
        # A negative offset on frame 0 would wrap round to the end of the
        # burst - a different instant entirely, not a missing one - so the
        # frames that cannot be shifted are dropped rather than read wrong.
        lo, hi = max(0, -min(offsets.values())), n_frames - 1 - max(0, max(offsets.values()))
        keep = [f for f in frames if lo <= f <= hi]
        if len(keep) != len(frames):
            print(f"offsets : {len(frames) - len(keep)} frame(s) dropped; with these "
                  f"offsets only {lo}..{hi} exist on every camera")
        frames = keep
    if not frames:
        raise SystemExit(f"no frame selected out of {n_frames}")

    ref = int(args.ref[1:]) if args.ref else min(
        cams, key=lambda i: (cam_rc(i)[0] - 1.5) ** 2 + (cam_rc(i)[1] - 1.5) ** 2)
    f_px = args.fpx * (args.scale if args.scale != 1.0 else 1.0)

    # Every calibrated warp is expressed in the reference camera's frame, so a
    # reference the file does not know sends the whole array back to the ideal
    # translation - worth saying out loud rather than quietly doing nothing.
    calibrated = []
    if calib is not None:
        calibrated = [i for i in sorted(cams) if relative_pose(calib, i, ref) is not None]
        fall_back = [cam_name(i) for i in sorted(cams) if i not in calibrated]
        print(f"calib   : {args.calib}  {len(calibrated)}/{len(cams)} camera(s) placed"
              + (f", ideal shift for {', '.join(fall_back)}" if fall_back else ""))
        if ref not in calib["cams"]:
            print(f"          {cam_name(ref)} is not in the calibration; the whole "
                  f"array falls back to the nominal grid. Pick --ref from "
                  f"{', '.join(cam_name(i) for i in sorted(calib['cams']))}")
        else:
            c = calib["cams"][ref]
            size = f"{c['size'][0]}x{c['size'][1]}" if c["size"] else "unstated size"
            print(f"          {cam_name(ref)} f = {c['K'][0, 0]:.0f}px at {size}, "
                  f"reprojection error {c['err']:.3f}px  "
                  f"(--fpx and --pitch now only serve the fallback)")
    # The scene arithmetic is only as good as f. Once --calib has placed the
    # reference camera, its measured focal length is the right one to use -
    # --fpx is a nominal 6mm/3.45um and was 3% out on this rig, which is 3% on
    # every derived window, range and limit below.
    f_scene = f_px
    if calib is not None and ref in calib["cams"]:
        c = calib["cams"][ref]
        f_scene = scaled_K(c["K"], c["size"], (1088, 1456))[0, 0] * args.scale
    sc = scene_scales(f_scene, args.occluder_mm, args.target_mm, args.target_size_mm,
                      args.pitch)
    level_win = int(3 * sc["target_px"]) | 1
    # CLAHE tiles a few target widths across: any finer and it equalises the
    # target itself away along with the shading it is there to open up.
    clahe_tile = max(2, int(round(1456 * args.scale / (3 * sc["target_px"]))))
    detect_px = sc["target_px"] if args.detect else 0.0
    detections = []
    # The flanking planes for the depth test: far enough that the fence's
    # aliased pattern reorganises, near enough that a target at the plane is
    # still a target at the plane. Half the depth of field either way does
    # both - a point at the centre moves half a target width across the
    # aperture, so its centroid does not move at all.
    detect_planes = ()
    print(f"scene   : target {args.target_size_mm:.0f}mm at {args.target_mm:.0f}mm "
          f"spans {sc['target_px']:.0f}px; one plane holds it inside its own width "
          f"from {sc['dof_near_mm']:.0f} to {sc['dof_far_mm']:.0f}mm")
    print(f"          fence at {args.occluder_mm:.0f}mm moves "
          f"{sc['fence_disparity_px']:.0f}px between neighbours and smears "
          f"{sc['fence_smear_px']:.0f}px across the {sc['span_mm']:.0f}mm aperture")
    if args.clahe > 0:
        print("          --clahe adapts per frame; it is for stills, and a sequence "
              "built on it will breathe")

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
    period = None
    if args.detect:
        half = 0.5 * (1.0 / sc["dof_near_mm"] - 1.0 / sc["dof_far_mm"]) / 2
        detect_planes = (1.0 / (1.0 / args.z + half), 1.0 / (1.0 / args.z - half))
    if args.sweep or args.notch_fence:
        probe = frames[len(frames) // 2]
        images, _, _ = load_all_cameras(cams, probe, args.scale, pool, offsets)
        images, _ = band_gain_normalise(images, cams_of_band)
        period, share = fence_period_px(images[ref])
        print(f"fence   : period {period:.1f}px in {cam_name(ref)} "
              f"({share * 100:.0f}% of the band), i.e. a mesh pitch of "
              f"{period * args.occluder_mm / f_scene:.1f}mm at {args.occluder_mm:.0f}mm")
    if args.sweep:
        planes = sweep_planes(args.sweep, sc, f_scene, args.pitch, args.scale)
        print(f"  {'z_mm':>9} {'disagreement':>14} {'fence phase':>12}  (1 = every view "
              f"sees the same phase, so the fence cannot cancel)")
        for zz in planes:
            warped = {i: warp_for_plane(images[i], i, ref, zz, f_px,
                                        args.pitch, calib)[0] for i in images}
            c = focus_cost(warped, cams_of_band, args.ksize,
                           None if args.keep_fence else period)
            ph = fence_phase_spread(sorted(images), ref, zz, f_scene, args.pitch,
                                    args.occluder_mm, period)
            costs.append((zz, c, ph))
            print(f"  {zz:9.1f} {c:14.4f} {ph:12.3f}")
            if args.save_sweep:
                comp, _, _, _ = band_composites(warped, cams_of_band, args.ksize,
                                                args.combine)
                d = out_dir / "sweep"
                d.mkdir(exist_ok=True)
                for band, img in comp.items():
                    write_png(d / f"band{band}_z{int(round(zz)):05d}.png", to_u8(img))
        z = min(costs, key=lambda zc: zc[1])[0]
        print(f"plane   : z = {z:.1f} mm (band-internal cost, frame {probe})")
        if z in (planes[0], planes[-1]) and len(planes) > 2:
            # The cost is not scale free across z: as the plane recedes every
            # warp shrinks towards the identity, so disagreement falls even
            # where nothing is in focus. A winner sitting on the end of the
            # range is that bias, not a plane.
            best = min((c for c in costs
                        if sc["dof_near_mm"] <= c[0] <= sc["dof_far_mm"]),
                       key=lambda zc: zc[1], default=None)
            print("          the cost was still falling at the end of the range, so "
                  "this is the sweep running out, not a focus peak.")
            if best:
                print(f"          inside the stated target range the best is "
                      f"z = {best[0]:.0f} mm (cost {best[1]:.4f}, fence phase "
                      f"{best[2]:.2f}); pick by the fence-phase column and by eye")
        (out_dir / "cost.csv").write_text(
            "z_mm,best_view_disagreement,fence_phase_resultant\n"
            + "".join(f"{a},{b},{c}\n" for a, b, c in costs))

    # ---- residual alignment, measured on the images ----------------------
    extra = {}
    if args.refine_from:
        saved = json.loads(args.refine_from.read_text())["shift_px"]
        extra = {int(k[1:]): tuple(v[:2]) for k, v in saved.items() if int(k[1:]) in cams}
        print(f"refine  : reusing {args.refine_from}")
    elif args.refine:
        probe = frames[len(frames) // 2]
        images, _, _ = load_all_cameras(cams, probe, args.scale, pool, offsets)
        images, _ = band_gain_normalise(images, cams_of_band)
        t0 = time.time()
        found = refine_alignment(images, cams_of_band, ref, z, f_px, args.pitch,
                                 calib, args.refine_ksize, args.refine_radius,
                                 args.refine_peak)
        print(f"refine  : frame {probe}, +-{args.refine_radius}px search, "
              f"{time.time() - t0:.1f}s")
        for i in sorted(found):
            dx, dy, peak, why = found[i]
            mark = f"  ({why}, left where it was)" if why and i != ref else ""
            print(f"          {cam_name(i)} ({cam_band(i)}nm)  "
                  f"dx={dx:+7.2f} dy={dy:+7.2f}  peak={peak:.3f}{mark}")
        moved = [i for i in found if i != ref and not found[i][3]]
        print(f"          {len(moved)} of {len(found) - 1} camera(s) moved")
        extra = {i: (v[0], v[1]) for i, v in found.items()}
        (out_dir / "refine.json").write_text(json.dumps({
            "z_mm": z, "frame": probe, "reference_camera": cam_name(ref),
            "radius_px": args.refine_radius, "min_peak": args.refine_peak,
            "scale": args.scale,
            "shift_px": {cam_name(i): list(found[i]) for i in sorted(found)},
        }, indent=2))

    # ---- the sequence ----------------------------------------------------
    # One 8-bit mapping for the whole sequence: recomputing it per frame makes
    # the video flicker with the scene's own brightness, which is exactly the
    # thing an SA result is meant to let you judge.
    stretch = {}
    written = {}
    t0 = time.time()
    for n, idx in enumerate(frames, 1):
        layers, comp, gains, shifts, found = process_frame(
            cams, idx, cams_of_band, ref, z, f_px, args.pitch, args.scale,
            args.ksize, calib, pool, extra, args.combine,
            period if args.notch_fence else None, level_win, detect_px,
            detect_planes, offsets)
        for f in found:
            detections.append({"frame": idx, **f})
        fc = false_colour(comp)
        if fc is not None:
            layers["falsecolour"] = fc

        for name, img in layers.items():
            if name == "capsules":
                base, found = img
                if args.stretch == "fixed" and name in stretch:
                    lo, hi = stretch[name]
                    u8 = to_u8(base, lo=lo, hi=hi)
                else:
                    u8, lo, hi = to_u8(base, return_range=True)
                    stretch.setdefault(name, (lo, hi))
                u8 = draw_capsules(tone_map(u8, args.gamma, args.clahe, clahe_tile),
                                   found)
            elif name == "falsecolour":
                u8 = tone_map(img, args.gamma, args.clahe, clahe_tile)
            elif name.startswith(("spread", "gloss")):
                # These fractions are already absolute: 0 is "every view
                # agreed", 1 is "all of the light". Stretching them on
                # percentiles would put the brightest thing in the frame at
                # white again, which is the very failure this layer had.
                # left untouched by --gamma/--clahe: these are absolute
                # fractions, and a display curve would stop them being read
                # off the pixel value.
                u8 = to_u8(img, lo=0.0, hi=1.0)
                stretch.setdefault(name, (0.0, 1.0))
            else:
                if args.stretch == "fixed" and name in stretch:
                    lo, hi = stretch[name]
                    u8 = to_u8(img, lo=lo, hi=hi)
                else:
                    u8, lo, hi = to_u8(img, return_range=True)
                    stretch.setdefault(name, (lo, hi))
                u8 = tone_map(u8, args.gamma, args.clahe, clahe_tile)
            d = out_dir / "png" / name
            d.mkdir(parents=True, exist_ok=True)
            write_png(d / f"frame{idx:04d}.png", u8)
            written[name] = written.get(name, 0) + 1
            if args.save_npy and name not in ("falsecolour", "capsules"):
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
        "sweep": [{"z_mm": a, "best_view_disagreement": b,
                   "fence_phase_resultant": c} for a, b, c in costs],
        "scene": {"occluder_mm": args.occluder_mm, "target_mm": args.target_mm,
                  "target_size_mm": args.target_size_mm,
                  "fence_period_px": period, **sc},
        "reference_camera": cam_name(ref),
        "calibration": str(args.calib) if args.calib else None,
        "calibrated_cameras": [cam_name(i) for i in calibrated],
        "combine": args.combine,
        "level_window_px": level_win,
        "gamma": args.gamma,
        "clahe": args.clahe,
        "detected_capsules": sum(1 for d in detections if not d["why"]),
        "refine_px": {cam_name(i): list(extra[i]) for i in sorted(extra)},
        "frame_offsets": {cam_name(i): offsets[i] for i in sorted(offsets)},
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
        "note": ("composites are per band and weighted by local agreement; "
                 "spread<nm> is the occlusion fraction in 0..1, not an IQR; the "
                 "cross-band image is locally normalised structure, not radiometry"),
    }, indent=2))

    if detections:
        kept = sum(1 for d in detections if not d["why"])
        (out_dir / "capsules.csv").write_text(
            "frame,x,y,major_px,minor_px,angle_deg,area_px,level,peak,gloss,"
            "rejected_because\n"
            + "".join(f"{d['frame']},{d['x']:.1f},{d['y']:.1f},{d['major']:.1f},"
                      f"{d['minor']:.1f},{d['angle']:.1f},{d['area']:.0f},"
                      f"{d['level']:.1f},{d['peak']:.1f},"
                      f"{d['gloss']:.3f},{d['why']}\n" for d in detections))
        print(f"capsules: {kept} kept, {len(detections) - kept} rejected over "
              f"{len(frames)} frame(s) -> capsules.csv")

    pool.shutdown()
    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\nwrote {sum(written.values())} image(s) in {len(written)} layer(s), "
          f"{total / 2**30:.2f} GiB, to {out_dir}")
    print(f"  png/band<nm>/          per-band composite ({args.combine} over that "
          f"band's views)")
    print("  png/spread<nm>/        occlusion fraction 0..1 - the share of the "
          "light\n                         that band's views lose to something in front")
    print("  png/gloss<nm>/         the bright side of the same disagreement -\n"
          "                         where a glossy target throws a view-dependent "
          "highlight")
    print("  png/crossband/         locally normalised structure across bands")
    if args.notch_fence:
        print("  png/defence<nm>/       the composite with the fence's own lattice "
              "notched out")
    if args.detect:
        print("  png/capsules/          the composite with target-sized blobs ringed: "
              "green\n                         kept, red rejected as a specular "
              "highlight")
    print("  png/falsecolour/       900/800/750 nm as R/G/B, each stretched alone")
    print("  png/bands_mosaic/      the four bands side by side")
    if videos:
        print("  video/<layer>.mp4      the same sequences as video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
