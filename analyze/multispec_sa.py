#!/usr/bin/env python3
"""
multispec_sa.py - Multispectral Synthetic Aperture imaging for fence removal.

Why a new SA is needed
----------------------
The 4x4 array carries a 2x2 bandpass tile that repeats over the whole grid:

        col0    col1    col2    col3
  row0  800nm   750nm   800nm   750nm      e00 e01 e02 e03
  row1  900nm   850nm   900nm   850nm      e04 e05 e06 e07
  row2  800nm   750nm   800nm   750nm      e08 e09 e10 e11
  row3  900nm   850nm   900nm   850nm      e12 e13 e14 e15

Classical synthetic aperture imaging assumes photo-consistency: a background
point projects to a *similar intensity* in every view, so variance / median /
entropy across views separates background (consistent) from occluder
(inconsistent).  Here that assumption is broken twice over:

  1. Each camera integrates a different 50nm slice of the NIR spectrum, so the
     radiance of the same surface point differs per camera by the material's
     spectral reflectance - an unknown, spatially varying factor.
  2. Filter transmission and scene illumination differ per band by up to ~10x
     (measured band means: 800nm 23-56, 750nm 41-111, 900nm 8-14, 850nm 16-37),
     so raw intensity is not even globally comparable.

This module therefore splits the problem along the axis where each cue is
still valid:

  * GEOMETRY  (alignment, plane sweep, depth) uses a *modality-invariant*
    representation - locally zero-mean, unit-variance normalised images
    (LMSN / per-pixel ZNCC descriptor).  This cancels any local affine
    photometric map I_b = a(x)*I_ref + b(x), which is exactly what a
    per-band reflectance factor looks like at local scale.
  * PHOTOMETRY (outlier rejection, compositing) is done strictly *within* a
    band, where the classical brightness-consistency assumption still holds.
    Each of the 4 bands has 4 cameras forming a 2x2 sub-aperture of 80mm
    pitch / 80mm extent.
  * HOLE FILLING uses the cross-band information again: pixels occluded in
    all 4 views of one band are recovered by transferring the structure of
    the all-band composite through a local affine spectral map.

The scene moves
---------------
The subject is a rotating tumbler mesh with capsules inside, so the scene is
NOT static during a burst and no image-domain temporal mean is admissible
(30-frame averaging erases every horizontal wire).  Time is instead treated as
a second aperture axis: the drum's own translation is estimated and undone, and
the neighbouring pulses join the 16 views as extra rays for the ROBUST
estimator.  The mesh sits at a different depth and radius, so it aligns under
neither transform and scatters twice over.

Run analyze/temporal_align.py first - the Pis are not NTP-synced, so the npz
timestamps cannot be used to index pulses across cameras.

Pipeline
--------
  prep     load 16 burst npz, extract a window of pulses at the aligned index,
           undistort, radiometric normalise, cache
  sweep    plane sweep on the centre pulse in the LMSN domain with a
           K-best-subset robust cost -> depth map + global cost curve
  defence  at the chosen plane: motion-compensated temporal ray stack ->
           iterative visibility (within-band residuals) -> visibility-weighted
           per-band composite -> cross-band hole fill; plus baselines
  report   false colour, band ratios, quantitative fence-suppression metrics

Usage
-----
  python analyze/temporal_align.py collected_data/20260728_072115
  python analyze/multispec_sa.py prep    collected_data/20260728_072115 \
      --center 45 --window 11 --prep-scale 0.5
  python analyze/multispec_sa.py sweep   collected_data/20260728_072115
  python analyze/multispec_sa.py defence collected_data/20260728_072115 --z-bg 600
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw10 import load_burst  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Array / filter layout
# ═══════════════════════════════════════════════════════════════════════════

GRID = (4, 4)
N_CAM = 16

#: 2x2 bandpass tile, repeated over the 4x4 grid.  Rows/cols as seen when the
#: array faces the subject; e00 is top-left, index increases along +col.
BAND_TILE = [[800, 750],
             [900, 850]]

#: camera index -> centre wavelength (nm)
CAM_BAND = {i: BAND_TILE[(i // 4) % 2][(i % 4) % 2] for i in range(N_CAM)}

#: wavelength -> list of camera indices
BANDS = {}
for _i, _b in CAM_BAND.items():
    BANDS.setdefault(_b, []).append(_i)
BAND_ORDER = sorted(BANDS)                       # [750, 800, 850, 900]

SAT_LEVEL = 940.0        # 10-bit, black level already subtracted (max ~963)


# ═══════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(npz_path: Path) -> dict:
    """Per-camera calibration -> K (16,3,3), dist (16,k), R (16,3,3), t (16,3) mm."""
    d = np.load(npz_path, allow_pickle=True)
    K = np.zeros((N_CAM, 3, 3))
    R = np.zeros((N_CAM, 3, 3))
    t = np.zeros((N_CAM, 3))
    dists = []
    for i in range(N_CAM):
        p = f"e{i:02d}"
        K[i] = d[f"{p}_camera_matrix"]
        dists.append(np.asarray(d[f"{p}_dist_coeffs"]).ravel())
        if f"{p}_board_rotation_matrix" in d:
            R[i] = d[f"{p}_board_rotation_matrix"]
        else:
            R[i], _ = cv2.Rodrigues(np.asarray(d[f"{p}_board_rvec"]).reshape(3))
        t[i] = np.asarray(d[f"{p}_board_tvec"]).reshape(3) * 1000.0  # m -> mm
    L = max(len(x) for x in dists)
    dist = np.zeros((N_CAM, L))
    for i, x in enumerate(dists):
        dist[i, :len(x)] = x
    image_size = np.asarray(d["e00_image_size"]).ravel().astype(int)  # (W, H)
    return dict(K=K, dist=dist, R=R, t=t, image_size=image_size)


def relative_poses(R: np.ndarray, t: np.ndarray, ref: int = 0):
    """ref -> i transform for every camera."""
    R_rel = np.zeros_like(R)
    t_rel = np.zeros_like(t)
    for i in range(R.shape[0]):
        R_rel[i] = R[i] @ R[ref].T
        t_rel[i] = t[i] - R_rel[i] @ t[ref]
    return R_rel, t_rel


def plane_homography(K_ref, K_i, R_rel, t_rel, z):
    """Homography mapping source camera i pixels -> reference, plane at depth z (mm)."""
    n = np.array([0.0, 0.0, 1.0])
    R_i2r = R_rel.T
    t_i2r = -R_rel.T @ t_rel
    return K_ref @ (R_i2r + np.outer(t_i2r, n) / float(z)) @ np.linalg.inv(K_i)


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    S = np.diag([s, s, 1.0])
    return S @ K


# ═══════════════════════════════════════════════════════════════════════════
# Multispectral photometry
# ═══════════════════════════════════════════════════════════════════════════

def robust_gain(img: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Per-camera radiometric scale.  Uses a high percentile of unsaturated
    pixels so that the (unknown) filter transmission x illumination product is
    divided out.  This makes *global* levels comparable; it does NOT make
    per-pixel intensities comparable across bands - that is what LMSN is for."""
    x = img if valid is None else img[valid]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    g = float(np.percentile(x, 98.0))
    return g if g > 1e-6 else 1.0


def local_norm(img: np.ndarray, ksize: int = 15, eps_frac: float = 0.25):
    """Locally zero-mean / unit-variance normalisation (LMSN).

    LN(x) = (I(x) - mu_w(x)) / (sigma_w(x) + eps)

    Any local affine photometric transform I' = a*I + b with a > 0 leaves LN
    unchanged.  A different bandpass changes the surface radiance by the
    material's spectral reflectance, which is locally close to a scale factor,
    so LN is the natural cross-band matching descriptor here.  eps is tied to
    the median local sigma so that flat, noise-dominated regions are not
    amplified into spurious structure.
    """
    img = img.astype(np.float32)
    k = (ksize, ksize)
    mu = cv2.boxFilter(img, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.boxFilter(img * img, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mu2 - mu * mu, 0.0)
    sig = np.sqrt(var)
    eps = max(1e-6, eps_frac * float(np.median(sig)))
    return (img - mu) / (sig + eps), mu, sig


def local_stats(img: np.ndarray, ksize: int = 15):
    k = (ksize, ksize)
    mu = cv2.boxFilter(img, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.boxFilter(img * img, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    sig = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    return mu, sig


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 - prep
# ═══════════════════════════════════════════════════════════════════════════

def find_burst(session: Path, cam: int, burst: int) -> Path:
    subs = sorted(session.glob(f"e{cam:02d}_*"))
    if not subs:
        raise FileNotFoundError(f"no directory for e{cam:02d} in {session}")
    cands = sorted(subs[0].glob(f"e{cam:02d}_burst{burst:04d}_*.npz"))
    if not cands:
        cands = sorted(subs[0].glob(f"e{cam:02d}_burst*.npz"))
    if not cands:
        raise FileNotFoundError(f"no burst npz for e{cam:02d}")
    return cands[0]


def load_offsets(out: Path, burst: int) -> dict:
    """Integer frame offsets from temporal_align.py (0 if not yet estimated).

    The Pis are not NTP-synced, so the per-frame `datetime.now()` stamps in the
    npz are unusable for cross-camera indexing - in session 20260728_072115
    they span 130 s.  The trigger, however, is wired straight to the camera
    boards, so the frames themselves ARE synchronous and only an integer index
    offset (measured: within +-3 frames of zero) is needed.
    """
    p = out / f"temporal_align_b{burst:04d}.json"
    if p.exists():
        d = json.loads(p.read_text())
        return {int(k[1:]): int(v) for k, v in d["offsets"].items()}
    print("  ! no temporal_align_b%04d.json - assuming zero offsets" % burst)
    return {i: 0 for i in range(N_CAM)}


def stage_prep(session: Path, out: Path, calib: Path, burst: int,
               center: int, window: int, scale: float):
    """Extract a TIME WINDOW of trigger pulses -> undistort -> cache.

    Deliberately NOT a temporal average.  The subject is a rotating tumbler
    mesh with capsules inside: content drifts ~2.4 px/frame, so averaging 30
    frames smears every horizontal wire away and leaves only the vertical bars
    (verified on e00).  The burst is instead kept as a sequence, because time
    is a second aperture axis - the mesh occludes a given background point in
    only SOME pulses, which is what makes a temporal ORDER STATISTIC (never a
    mean) able to reject it where 4 views per band alone cannot.
    """
    cal = load_calibration(calib)
    W, H = int(cal["image_size"][0]), int(cal["image_size"][1])
    out.mkdir(parents=True, exist_ok=True)

    from raw10 import unpack_raw10

    off = load_offsets(out, burst)
    half = window // 2
    T = 2 * half + 1
    hs, ws = (int(H * scale), int(W * scale)) if scale != 1.0 else (H, W)

    seq = np.zeros((N_CAM, T, hs, ws), np.float16)
    valid = np.zeros((N_CAM, hs, ws), np.float32)
    gains = np.zeros(N_CAM)
    newKs = np.zeros((N_CAM, 3, 3))

    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    for i in range(N_CAM):
        cp = cache / f"e{i:02d}_seq_b{burst:04d}_c{center}_w{T}_s{scale}.npz"
        if cp.exists():
            cd = np.load(cp)
            seq[i], valid[i] = cd["seq"], cd["valid"]
            gains[i], newKs[i] = float(cd["gain"]), cd["newK"]
            print(f"  e{i:02d} {CAM_BAND[i]}nm  (cached)", flush=True)
            continue
        p = find_burst(session, i, burst)
        d = np.load(p)
        packed = d["frames_raw"]
        bl = float(np.asarray(d["black_levels"]).ravel()[0]) / 64.0
        n_avail = packed.shape[0]
        # same physical pulse in every camera == same index + that camera's offset
        idx = np.clip(np.arange(center - half, center + half + 1) + off[i],
                      0, n_avail - 1)
        fr = unpack_raw10(packed[idx], W).astype(np.float32)
        fr = np.maximum(fr - bl, 0.0)
        del packed, d

        sat = (fr.max(axis=0) >= SAT_LEVEL)
        gains[i] = robust_gain(fr[half], ~sat)

        newK, _ = cv2.getOptimalNewCameraMatrix(cal["K"][i], cal["dist"][i],
                                                (W, H), 0, (W, H))
        newKs[i] = newK
        ok = cv2.undistort((~sat).astype(np.float32), cal["K"][i],
                           cal["dist"][i], None, newK)
        if scale != 1.0:
            ok = cv2.resize(ok, (ws, hs), interpolation=cv2.INTER_AREA)
        valid[i] = (ok > 0.99).astype(np.float32)

        for k in range(T):
            u = cv2.undistort(fr[k], cal["K"][i], cal["dist"][i], None, newK)
            if scale != 1.0:
                u = cv2.resize(u, (ws, hs), interpolation=cv2.INTER_AREA)
            seq[i, k] = (u / gains[i]).astype(np.float16)
        del fr
        np.savez(cp, seq=seq[i], valid=valid[i], gain=gains[i], newK=newK)
        print(f"  e{i:02d} {CAM_BAND[i]}nm  pulses={idx[0]}..{idx[-1]} "
              f"(off {off[i]:+d})  gain={gains[i]:8.2f}  "
              f"sat={100*sat.mean():5.2f}%", flush=True)

    cal["K"] = np.stack([scale_K(k, scale) for k in newKs]) if scale != 1.0 else newKs

    np.savez(out / "prep.npz",
             seq=seq, valid=valid, gains=gains,
             K=cal["K"], R=cal["R"], t=cal["t"],
             bands=np.array([CAM_BAND[i] for i in range(N_CAM)]),
             center=center, window=T, offsets=np.array([off[i] for i in range(N_CAM)]),
             scale=scale)
    print(f"[prep] wrote {out/'prep.npz'}  seq={seq.shape} (N, T, H, W)")

    mosaic = _mosaic(seq[:, half].astype(np.float32),
                     [f"e{i:02d} {CAM_BAND[i]}nm" for i in range(N_CAM)])
    cv2.imwrite(str(out / "00_input_mosaic.png"), mosaic)


def _tonemap(x: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros(x.shape, np.uint8)
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    y = np.clip((x - a) / max(b - a, 1e-6), 0, 1) ** (1 / 1.8)
    return (y * 255).astype(np.uint8)


def _mosaic(stack, labels, cols=4, tile_w=364):
    tiles = []
    for i, im in enumerate(stack):
        t = _tonemap(im)
        h = int(t.shape[0] * tile_w / t.shape[1])
        t = cv2.resize(t, (tile_w, h))
        cv2.putText(t, labels[i], (8, 24), 0, 0.6, 255, 2)
        tiles.append(t)
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Warping
# ═══════════════════════════════════════════════════════════════════════════

def warp_stack(stack, valid, K, R_rel, t_rel, z, ref=0):
    """Warp all views to the reference view for a fronto-parallel plane at z."""
    N, H, W = stack.shape
    out = np.zeros_like(stack)
    msk = np.zeros_like(stack)
    for i in range(N):
        if i == ref:
            out[i] = stack[i]
            msk[i] = valid[i]
            continue
        Hm = plane_homography(K[ref], K[i], R_rel[i], t_rel[i], z)
        out[i] = cv2.warpPerspective(stack[i], Hm, (W, H),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        msk[i] = cv2.warpPerspective(valid[i], Hm, (W, H),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, msk


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 - plane sweep in the modality-invariant domain
# ═══════════════════════════════════════════════════════════════════════════

def kbest_cost(L, M, k):
    """K-best-subset photo-consistency cost on the LMSN stack.

    At each pixel the residual to the across-view median is computed, the k
    smallest residuals are kept and averaged.  Occluded views (the mesh) are
    simply not in the k-subset, so the cost stays low for a background point
    even when a majority of views are blocked - the failure mode that plain
    variance and even the median suffer from at high occlusion ratios.
    """
    X = np.where(M > 0.5, L, np.nan)
    with warnings.catch_warnings():
        # pixels outside every view's frustum give all-NaN slices; they are
        # assigned the sentinel cost below.
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X, axis=0)
        r = np.abs(X - med)
        r = np.where(np.isfinite(r), r, np.inf)
        r.sort(axis=0)
        kk = np.where(np.isfinite(r[:k]), r[:k], np.nan)
        c = np.nanmean(kk, axis=0)
    return np.where(np.isfinite(c), c, 10.0).astype(np.float32)


def stage_sweep(out: Path, z_near: float, z_far: float, n_planes: int,
                sweep_scale: float, k_best: int, ln_ksize: int, ref: int):
    d = np.load(out / "prep.npz")
    # centre pulse only - the sweep must see an unsmeared instant of the drum
    imgs = d["seq"][:, int(d["window"]) // 2].astype(np.float32)
    valid = d["valid"]
    K, R, t = d["K"].copy(), d["R"], d["t"]
    N, H, W = imgs.shape

    if sweep_scale != 1.0:
        h2, w2 = int(H * sweep_scale), int(W * sweep_scale)
        imgs = np.stack([cv2.resize(x, (w2, h2), interpolation=cv2.INTER_AREA) for x in imgs])
        valid = np.stack([cv2.resize(x, (w2, h2), interpolation=cv2.INTER_NEAREST) for x in valid])
        K = np.stack([scale_K(k, sweep_scale) for k in K])
        H, W = h2, w2

    # modality-invariant descriptor, computed once per view in its own frame
    LN = np.stack([local_norm(imgs[i], ln_ksize)[0] for i in range(N)])
    R_rel, t_rel = relative_poses(R, t, ref)

    zs = 1.0 / np.linspace(1.0 / z_far, 1.0 / z_near, n_planes)
    cost = np.zeros((n_planes, H, W), np.float32)
    stack_prev = []
    for j, z in enumerate(zs):
        Lw, Mw = warp_stack(LN, valid, K, R_rel, t_rel, z, ref)
        cost[j] = kbest_cost(Lw, Mw, k_best)
        if j % 8 == 0:
            print(f"  plane {j:3d}/{n_planes}  z={z:8.1f} mm  "
                  f"cost={cost[j].mean():.4f}", flush=True)

    depth_idx = np.argmin(cost, axis=0)
    depth = zs[depth_idx].astype(np.float32)
    curve = cost.reshape(n_planes, -1).mean(axis=1)

    np.savez_compressed(out / "sweep.npz", zs=zs, curve=curve,
                        depth=depth, cost_min=cost.min(axis=0),
                        sweep_scale=sweep_scale)

    # visualisation
    dv = np.clip((1.0 / depth - 1.0 / z_far) / (1.0 / z_near - 1.0 / z_far), 0, 1)
    cv2.imwrite(str(out / "10_depth_disparity.png"),
                cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    _plot_curve(zs, curve, out / "11_sweep_cost.png")
    print(f"[sweep] best global plane z = {zs[int(np.argmin(curve))]:.1f} mm")
    return zs, curve, depth


def _plot_curve(zs, curve, path):
    Wp, Hp = 900, 420
    img = np.full((Hp, Wp, 3), 255, np.uint8)
    x = np.log(zs)
    x = (x - x.min()) / max(float(np.ptp(x)), 1e-9)
    y = (curve - curve.min()) / max(float(np.ptp(curve)), 1e-9)
    pts = np.stack([(60 + x * (Wp - 90)), (Hp - 60 - y * (Hp - 110))], 1).astype(int)
    cv2.polylines(img, [pts], False, (200, 60, 40), 2)
    for i in range(0, len(zs), max(1, len(zs) // 8)):
        cv2.putText(img, f"{zs[i]:.0f}", (pts[i, 0] - 18, Hp - 30), 0, 0.45, (0, 0, 0), 1)
        cv2.line(img, (pts[i, 0], Hp - 60), (pts[i, 0], Hp - 52), (0, 0, 0), 1)
    cv2.putText(img, "plane-sweep cost (LMSN, k-best subset)  vs  depth [mm]",
                (60, 34), 0, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 - multispectral robust compositing
# ═══════════════════════════════════════════════════════════════════════════

def tukey_w(r, c):
    u = np.clip(r / np.maximum(c, 1e-9), 0, 1)
    return (1 - u * u) ** 2


def multispectral_composite(Vw, Mw, bands, ln_ksize=15, n_iter=4,
                            tukey_c=2.5, min_views=1):
    """Visibility-weighted, band-wise robust background estimation.

    Each view is compared ONLY against the running estimate of its own band, so
    the residual test never has to explain away a spectral reflectance
    difference - only occlusion and noise.  Visibility is then shared across
    bands (an occluder is an occluder in every band, at the same warped
    position), which restores the statistical power of all 16 views to the
    detection step while keeping the photometry band-pure.
    """
    N, H, W = Vw.shape
    band_ids = sorted(set(bands))
    vis = Mw.copy()

    # init: per-band median of the available views
    comp = {}
    for b in band_ids:
        idx = [i for i in range(N) if bands[i] == b]
        X = np.where(Mw[idx] > 0.5, Vw[idx], np.nan)
        with np.errstate(invalid="ignore"):
            comp[b] = np.nan_to_num(np.nanmedian(X, axis=0))

    for it in range(n_iter):
        # ── residuals against own-band estimate ───────────────────────────
        res = np.zeros((N, H, W), np.float32)
        for i in range(N):
            res[i] = Vw[i] - comp[bands[i]]
        # robust scale per band (MAD over valid pixels of that band)
        scale = {}
        for b in band_ids:
            idx = [i for i in range(N) if bands[i] == b]
            r = res[idx][Mw[idx] > 0.5]
            s = 1.4826 * np.median(np.abs(r - np.median(r))) if r.size else 1e-3
            scale[b] = max(float(s), 1e-4)

        # ── shared visibility: normalised residual, pooled over bands ────
        rn = np.stack([np.abs(res[i]) / scale[bands[i]] for i in range(N)])
        w = tukey_w(rn, tukey_c) * Mw
        # an occluder in front is *darker or brighter consistently*; keep the
        # soft weight but hard-reject the worst quartile when enough views
        # remain, which is what gives the sharp mesh cut
        vis = w

        # ── band-wise weighted composite ──────────────────────────────────
        newcomp = {}
        cover = {}
        for b in band_ids:
            idx = [i for i in range(N) if bands[i] == b]
            ws = vis[idx].sum(axis=0)
            acc = (vis[idx] * Vw[idx]).sum(axis=0)
            ok = ws > (min_views * 1e-3)
            est = np.where(ok, acc / np.maximum(ws, 1e-9), comp[b])
            newcomp[b] = est.astype(np.float32)
            cover[b] = ws.astype(np.float32)
        comp = newcomp

    return comp, cover, vis


def cross_band_fill(comp, cover, ln_ksize=21, cover_thresh=0.25):
    """Recover pixels that every view of one band lost to the mesh.

    A pooled all-band estimate is built from whichever bands still have
    coverage, and transferred into the deficient band through a LOCAL AFFINE
    spectral map  I_b ~ a(x)*P(x) + b(x), with a,b regressed from the
    surrounding well-covered neighbourhood.  This is the step that only exists
    because the array is multispectral: a single-band SA has nothing to fall
    back on and must inpaint from spatial context alone.
    """
    band_ids = sorted(comp)
    C = np.stack([cover[b] for b in band_ids])
    Cn = C / np.maximum(C.max(axis=(1, 2), keepdims=True), 1e-9)
    P_num = np.zeros_like(next(iter(comp.values())))
    P_den = np.zeros_like(P_num)
    # pooled structure: each band standardised, weighted by its coverage
    for j, b in enumerate(band_ids):
        mu, sig = local_stats(comp[b], ln_ksize)
        z = (comp[b] - mu) / (sig + 1e-4)
        P_num += Cn[j] * z
        P_den += Cn[j]
    P = P_num / np.maximum(P_den, 1e-9)

    out = {}
    filled = {}
    for j, b in enumerate(band_ids):
        good = (Cn[j] > cover_thresh).astype(np.float32)
        Ib = comp[b]
        # local affine regression of Ib on P, using only well-covered pixels
        k = (ln_ksize, ln_ksize)
        bf = lambda x: cv2.boxFilter(x * good, -1, k, normalize=False,
                                     borderType=cv2.BORDER_REFLECT)
        n = cv2.boxFilter(good, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
        n = np.maximum(n, 1e-6)
        mP, mI = bf(P) / n, bf(Ib) / n
        vP = bf(P * P) / n - mP * mP
        cPI = bf(P * Ib) / n - mP * mI
        a = cPI / np.maximum(vP, 1e-6)
        c = mI - a * mP
        pred = a * P + c
        alpha = np.clip((Cn[j] - 0.05) / max(cover_thresh - 0.05, 1e-6), 0, 1)
        out[b] = (alpha * Ib + (1 - alpha) * pred).astype(np.float32)
        filled[b] = (1 - alpha).astype(np.float32)
    return out, filled


def estimate_scene_motion(seq_ref: np.ndarray, half: int):
    """Per-pulse global translation of the drum content, in the reference view.

    The tumbler rotates, so a background point does NOT stay at the same pixel
    from one pulse to the next - pooling over time without compensating would
    smear exactly what we are trying to measure.  Translation is estimated by
    phase correlation on a strongly low-passed, locally normalised image: the
    low-pass is essential because the bare mesh is a near-periodic texture and
    raw phase correlation locks onto the wrong harmonic (measured |dx| of
    60-165 px on frames whose true motion is ~2 px).

    Returns (T, 2) float array of (dx, dy) relative to the centre pulse.
    """
    T = seq_ref.shape[0]
    prep = []
    win = None
    for k in range(T):
        f = seq_ref[k].astype(np.float32)
        g = f - cv2.blur(f, (31, 31))
        g = cv2.GaussianBlur(g, (0, 0), 3.0)      # kill the mesh harmonics
        g = g / (g.std() + 1e-9)
        if win is None:
            win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        prep.append((g * win).astype(np.float64))
    mv = np.zeros((T, 2))
    for k in range(T):
        if k != half:
            (dx, dy), _ = cv2.phaseCorrelate(prep[half], prep[k])
            mv[k] = (dx, dy)
    # a rigid drum turns at a near-constant rate: fit a line through the
    # per-pulse estimates (IRLS) so a single bad lock cannot corrupt the stack
    ks = np.arange(T) - half
    A = np.stack([ks, np.ones(T)], 1)
    for c in range(2):
        w = np.ones(T)
        sol = np.zeros(2)
        for _ in range(3):
            sol, *_ = np.linalg.lstsq(A * w[:, None], mv[:, c] * w, rcond=None)
            r = np.abs(A @ sol - mv[:, c])
            s = 1.4826 * np.median(r) + 1e-6
            w = (r < 2.5 * s).astype(float)
        mv[:, c] = A @ sol
    return mv


def stage_defence(out: Path, z_bg: float, ln_ksize: int, n_iter: int,
                  tukey_c: float, ref: int, work_scale: float,
                  use_time: bool = True):
    d = np.load(out / "prep.npz")
    seq, valid = d["seq"], d["valid"]
    K, R, t = d["K"].copy(), d["R"], d["t"]
    bands = list(d["bands"])
    T = int(d["window"])
    half = T // 2
    N, _, H, W = seq.shape

    if work_scale != 1.0:
        h2, w2 = int(H * work_scale), int(W * work_scale)
        seq = np.stack([[cv2.resize(seq[i, k].astype(np.float32), (w2, h2),
                                    interpolation=cv2.INTER_AREA)
                         for k in range(T)] for i in range(N)]).astype(np.float16)
        valid = np.stack([cv2.resize(x, (w2, h2), interpolation=cv2.INTER_NEAREST)
                          for x in valid])
        K = np.stack([scale_K(k, work_scale) for k in K])
        H, W = h2, w2

    R_rel, t_rel = relative_poses(R, t, ref)
    imgs = seq[:, half].astype(np.float32)

    # ── build the ray sample stack ───────────────────────────────────────
    # 16 cameras at the centre pulse, extended with the neighbouring pulses
    # after compensating the drum's own motion.  A within-band median has only
    # 4 samples from the array alone; +-half pulses takes that to 4*T, which is
    # what makes a band-pure order statistic usable at this occlusion ratio.
    Vw, Mw = warp_stack(imgs, valid, K, R_rel, t_rel, z_bg, ref)
    samp_band = list(bands)

    if use_time and T > 1:
        mv = estimate_scene_motion(seq[ref].astype(np.float32), half)
        step = mv[half + 1] - mv[half] if T > half + 1 else mv[half] - mv[half - 1]
        print(f"[defence] drum motion at ref = "
              f"({step[0]:+.2f}, {step[1]:+.2f}) px/pulse")
        np.save(out / "scene_motion.npy", mv)
        eV, eM, eb = [], [], []
        for k in range(T):
            if k == half:
                continue
            Vk, Mk = warp_stack(seq[:, k].astype(np.float32), valid, K,
                                R_rel, t_rel, z_bg, ref)
            # undo the drum translation so pulse k lands on pulse `half`.
            # The mesh sits at a different depth AND a different radius, so it
            # does NOT align under this transform - it scatters in time as well
            # as across the aperture, which is the whole point.
            A = np.float32([[1, 0, -mv[k, 0]], [0, 1, -mv[k, 1]]])
            for i in range(N):
                eV.append(cv2.warpAffine(Vk[i], A, (W, H),
                                         flags=cv2.INTER_LINEAR, borderValue=0))
                eM.append(cv2.warpAffine(Mk[i], A, (W, H),
                                         flags=cv2.INTER_NEAREST, borderValue=0))
                eb.append(bands[i])
        Vw = np.concatenate([Vw, np.stack(eV)], 0)
        Mw = np.concatenate([Mw, np.stack(eM)], 0)
        samp_band = samp_band + eb
        del eV, eM
        print(f"[defence] sample stack: {Vw.shape[0]} rays "
              f"({N} views x {T} pulses)")

    comp, cover, vis = multispectral_composite(
        Vw, Mw, samp_band, ln_ksize=ln_ksize, n_iter=n_iter, tukey_c=tukey_c)
    final, filled = cross_band_fill(comp, cover, ln_ksize=max(21, ln_ksize))

    # ── baselines for comparison ─────────────────────────────────────────
    # all baselines use ONLY the 16 centre-pulse rays, so the comparison
    # isolates what the multispectral treatment buys over classical SA on the
    # same data; the temporal extension is reported separately.
    V0, M0 = Vw[:N], Mw[:N]
    base_mean = (V0 * M0).sum(0) / np.maximum(M0.sum(0), 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_med16 = np.nan_to_num(
            np.nanmedian(np.where(M0 > 0.5, V0, np.nan), axis=0))
        single = {}
        for b in sorted(set(bands)):
            idx = [i for i in range(N) if bands[i] == b]
            single[b] = np.nan_to_num(
                np.nanmedian(np.where(M0[idx] > 0.5, V0[idx], np.nan), axis=0))

    np.savez_compressed(
        out / "defence.npz", z_bg=z_bg,
        **{f"final_{b}": final[b] for b in final},
        **{f"comp_{b}": comp[b] for b in comp},
        **{f"cover_{b}": cover[b] for b in cover},
        **{f"single_{b}": single[b] for b in single},
        base_mean=base_mean, base_med16=base_med16,
        ref_view=Vw[ref], vis_mean=vis.mean(0),
    )

    # ── outputs ──────────────────────────────────────────────────────────
    for b in final:
        cv2.imwrite(str(out / f"20_defenced_{b}nm.png"), _tonemap(final[b]))
        cv2.imwrite(str(out / f"21_coverage_{b}nm.png"),
                    cv2.applyColorMap(
                        _tonemap(cover[b] / max(cover[b].max(), 1e-9), 0, 100),
                        cv2.COLORMAP_VIRIDIS))
    cv2.imwrite(str(out / "22_input_ref.png"), _tonemap(imgs[ref]))
    cv2.imwrite(str(out / "23_baseline_mean.png"), _tonemap(base_mean))
    cv2.imwrite(str(out / "24_baseline_median16.png"), _tonemap(base_med16))

    fc = false_colour(final)
    cv2.imwrite(str(out / "30_false_colour.png"), fc)

    cmp_img = _mosaic(
        np.stack([imgs[ref], base_mean, base_med16,
                  single[BAND_ORDER[1]], final[BAND_ORDER[1]], final[BAND_ORDER[3]]]),
        ["input e%02d" % ref, "SA mean (all 16)", "SA median (band-blind)",
         f"single-band SA {BAND_ORDER[1]}nm",
         f"multispec SA {BAND_ORDER[1]}nm", f"multispec SA {BAND_ORDER[3]}nm"],
        cols=3, tile_w=460)
    cv2.imwrite(str(out / "31_comparison.png"), cmp_img)

    metrics = evaluate(imgs[ref], {"SA mean": base_mean,
                                   "SA median (band-blind)": base_med16,
                                   f"single-band SA {BAND_ORDER[1]}nm": single[BAND_ORDER[1]],
                                   f"multispec SA {BAND_ORDER[1]}nm": final[BAND_ORDER[1]]})
    metrics["z_bg_mm"] = float(z_bg)
    metrics["band_map"] = {f"e{i:02d}": int(CAM_BAND[i]) for i in range(N_CAM)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return final


def false_colour(final):
    """R=900nm, G=850nm, B=750nm after per-band percentile stretch."""
    def ch(b):
        return _tonemap(final[b]).astype(np.float32)
    r, g, bl = ch(900), ch(850), ch(750)
    return np.stack([bl, g, r], -1).astype(np.uint8)   # BGR for imwrite


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def mesh_peaks(img, n_peaks=12, guard=6):
    """Locate the periodic mesh harmonics in the reference view's spectrum."""
    x = img - img.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(x * win)))
    cy, cx = np.array(F.shape) // 2
    F[cy - guard:cy + guard + 1, cx - guard:cx + guard + 1] = 0
    flat = F.ravel().copy()
    peaks = []
    for _ in range(n_peaks):
        k = int(np.argmax(flat))
        py, px = divmod(k, F.shape[1])
        peaks.append((py, px))
        y0, y1 = max(0, py - guard), min(F.shape[0], py + guard + 1)
        x0, x1 = max(0, px - guard), min(F.shape[1], px + guard + 1)
        tmp = flat.reshape(F.shape)
        tmp[y0:y1, x0:x1] = 0
        flat = tmp.ravel()
    return peaks, F.shape


def peak_energy(img, peaks, guard=4):
    x = img - img.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    e = 0.0
    for py, px in peaks:
        y0, y1 = max(0, py - guard), min(F.shape[0], py + guard + 1)
        x0, x1 = max(0, px - guard), min(F.shape[1], px + guard + 1)
        e += float(F[y0:y1, x0:x1].sum())
    return e / max(float(F.sum()), 1e-12)


def evaluate(ref_img, results: dict):
    """Fence Suppression Ratio at the mesh harmonics + residual high-freq energy."""
    peaks, _ = mesh_peaks(ref_img)
    e_ref = peak_energy(ref_img, peaks)
    lap_ref = float(np.abs(cv2.Laplacian(ref_img, cv2.CV_32F)).mean())
    m = {"reference": {"mesh_peak_energy_frac": e_ref, "mean_abs_laplacian": lap_ref}}
    for name, img in results.items():
        e = peak_energy(img, peaks)
        lap = float(np.abs(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)).mean())
        m[name] = {
            "mesh_peak_energy_frac": e,
            "FSR_dB": float(10 * np.log10(max(e_ref, 1e-15) / max(e, 1e-15))),
            "mean_abs_laplacian": lap,
            "detail_retention": float(lap / max(lap_ref, 1e-9)),
        }
    return m


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "sweep", "defence", "all"])
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--calib", type=Path, default=Path("calibration_results.npz"))
    ap.add_argument("--burst", type=int, default=1)
    ap.add_argument("--center", type=int, default=45,
                    help="index of the trigger pulse to reconstruct")
    ap.add_argument("--window", type=int, default=11,
                    help="pulses kept around --center (time = 2nd aperture axis)")
    ap.add_argument("--no-time", action="store_true",
                    help="composite from the 16 centre-pulse rays only")
    ap.add_argument("--prep-scale", type=float, default=1.0)
    ap.add_argument("--sweep-scale", type=float, default=0.35)
    ap.add_argument("--work-scale", type=float, default=1.0)
    # drum is ~1 m across: the near mesh wall sits at ~<=100 mm and the far
    # wall around 1.1 m, so the capsules lie somewhere inside that span
    ap.add_argument("--z-near", type=float, default=60.0)
    ap.add_argument("--z-far", type=float, default=1600.0)
    ap.add_argument("--planes", type=int, default=64)
    ap.add_argument("--k-best", type=int, default=6)
    ap.add_argument("--ln", type=int, default=15)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--tukey-c", type=float, default=2.5)
    ap.add_argument("--ref", type=int, default=0)
    ap.add_argument("--z-bg", type=float, default=None)
    a = ap.parse_args()

    out = a.out or (a.session.parent.parent / "sa_results" / a.session.name)
    out.mkdir(parents=True, exist_ok=True)

    if a.stage in ("prep", "all"):
        print("[prep]")
        stage_prep(a.session, out, a.calib, a.burst, a.center, a.window,
                   a.prep_scale)
    if a.stage in ("sweep", "all"):
        print("[sweep]")
        stage_sweep(out, a.z_near, a.z_far, a.planes, a.sweep_scale,
                    a.k_best, a.ln, a.ref)
    if a.stage in ("defence", "all"):
        z = a.z_bg
        if z is None:
            s = np.load(out / "sweep.npz")
            z = float(s["zs"][int(np.argmin(s["curve"]))])
            print(f"[defence] auto z_bg = {z:.1f} mm")
        print("[defence]")
        stage_defence(out, z, a.ln, a.iters, a.tukey_c, a.ref, a.work_scale,
                      use_time=not a.no_time)


if __name__ == "__main__":
    main()
