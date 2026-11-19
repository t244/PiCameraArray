#!/usr/bin/env python3
"""
sa_defence.py - de-fencing one band by decomposing it, not by masking it.

The array gives four views per band. That is too few for a median to
reconstruct anything: at four samples the median is the average of the middle
two, so an occluder covering two views is half of the answer. And the occluder
here does not cover pixels in the first place - the tumbler mesh sits about
100 mm from the lens while the focus is at 400-700 mm, so it is defocused to
roughly its own period and roughly half open. It attenuates EVERY view by a
different amount at every point.

That makes this a component decomposition, not a segmentation:

    I_i(x) = T_i(x) . B(x)          T_i in (0, 1],  B common to all views

B is the background - the thing being looked at - and T_i is the transmission
the fence leaves in view i. What makes it solvable with only four views is
that 40 mm of baseline slides the fence by hundreds of pixels while the
background stays put: T_i is wildly different between views, B is the same in
all of them. View diversity is the whole leverage, so everything here is built
to preserve it.

WHY A GLOBAL WARP IS NOT ENOUGH
-------------------------------
calibration_results.npz came off a different prototype. Its intrinsics are
close enough to place the cameras and remove most of the distortion, and the
plane-induced homography built from it takes out the bulk of the parallax -
hundreds of pixels of it. What it leaves is a residual that VARIES ACROSS THE
FRAME, because a distortion model that is a few percent wrong is wrong by a
different amount at the centre than at the corner. One shift per view cannot
absorb that. So the alignment here is two-stage: the calibration globally,
then a block-wise match locally, warped by the flow it produces.

THE STEPS
---------
  1 warp      the plane-induced homography, per view              (sa_packed)
  2 mask      per view, loosely, where the fence takes light and where the
              target throws a specular flash. Both are excluded from the
              matching - the fence because it is not the thing being matched,
              the highlight because it moves with the viewpoint and would pull
              the match onto a feature that is not there in the other views
  3 match     block-wise masked correlation against the reference view, giving
              a displacement per block, regularised and upsampled to a flow
  4 warp      each view by its own flow, locally
  5 separate  alternate between B and T_i until they stop moving
  6 fill      where every view was blocked there is nothing to recover, so
              the hole is synthesised from the texture around it

Each stage writes an image, because the only way to tell which stage broke is
to look at all of them.

WHICH PLANE
-----------
--sweep runs every plane over the range the target occupies, uniform in
disparity, and reports the fence's share of the band at each alongside the
fence-phase prediction: a purely geometric number saying whether the views see
the fence at different enough phases for an average to cancel it at all.

Measured on burst0001 at 800 nm over 17 planes, the median composite's fence
share tracks that prediction at r = +0.70 - it runs from 6.2% where the phases
are spread to 11.7% where they coincide. The decomposition stays between 5.3%
and 6.0% throughout. Its advantage doubles, from 2.0 points to 4.1, exactly
where the median has nothing to work with, which is the point of doing the
decomposition rather than an average.

USAGE
-----
    python analyze/sa_defence.py <burst> --calib calibration_results.npz \\
        --band 800 --z 550 --frame 45
    python analyze/sa_defence.py <burst> --calib calibration_results.npz \\
        --band 800 --sweep --gamma 1.6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sa_packed as S                                   # noqa: E402
from sa_packed import cv2                               # noqa: E402


# ------------------------------------------------------------------ masking --
def upper_envelope(img: np.ndarray, period: float):
    """What the view would read here if the fence were not in the way.

    A dilation over rather more than one fence period reaches past the bars to
    the brightest thing in the neighbourhood, which - the fence being roughly
    half open - is a place the fence was not.

    It has to be a dilation of a SMOOTHED image, though. Raw, the maximum in
    any window is a specular glint off a capsule, so the envelope becomes a
    map of the glints and every ordinary pixel reads as though the fence were
    almost shut: measured on burst0001, only 5% of a view came out as
    matchable. Smoothing first, at a fraction of the target, takes the
    isolated spikes out and leaves the level. It also puts the envelope on the
    same footing as what it is compared against, which matters, because the
    fence is defocused and its attenuation is a smooth field too.
    """
    k = int(max(3, round(period * 1.3))) | 1
    if cv2 is not None:
        a = cv2.GaussianBlur(np.nan_to_num(img), (0, 0), max(1.0, period * 0.25))
        # The envelope is low-frequency by construction and is blurred again
        # straight after, so the dilation runs on a quarter-size copy. An
        # elliptical structuring element is not separable, and at full size a
        # 40-pixel one costs more than every other stage of a plane together.
        f = 4 if min(a.shape) > 8 * k else 1
        small = (cv2.resize(a, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
                 if f > 1 else a)
        ks = int(max(3, round(k / f))) | 1
        env = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        if f > 1:
            env = cv2.resize(env, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(env, (0, 0), period * 0.5)
    a = np.nan_to_num(img)
    r = k // 2
    pad = np.pad(a, r, mode="edge")
    out = np.zeros_like(a)
    for dy in range(-r, r + 1, max(1, r // 3)):
        for dx in range(-r, r + 1, max(1, r // 3)):
            out = np.maximum(out, pad[r + dy:r + dy + a.shape[0],
                                      r + dx:r + dx + a.shape[1]])
    return out


def view_masks(img: np.ndarray, period: float, band=(90.0, 99.0),
               open_at=0.55):
    """(transmission, matching weight) for one view.

    transmission  img / its own upper envelope, in 0..1. Low where the fence
                  takes light. It is an estimate of T_i before anything is
                  solved, and it is only used to decide what to trust.
    weight        what the matcher is allowed to look at: the parts that are
                  reasonably clear of the fence, are not clipped, and are not
                  a specular flash.

    The highlight exclusion matters as much as the fence one. The capsules are
    glossy ellipsoids, so a highlight sits at a different place on the capsule
    for every viewpoint. Matching on one is matching on something that is not
    in the other view at all, and it will drag the block somewhere wrong with
    a confident correlation score.
    """
    a = np.nan_to_num(img).astype(np.float32)
    finite = np.isfinite(img)
    env = upper_envelope(a, period)
    # compared smoothed against smoothed: the transmission of a defocused
    # fence is a smooth field, and dividing the raw image by it would just
    # push the scene's own texture into the mask
    sm = (cv2.GaussianBlur(a, (0, 0), max(1.0, period * 0.25))
          if cv2 is not None else a)
    with np.errstate(invalid="ignore", divide="ignore"):
        trans = sm / np.maximum(env, 1e-6)
    trans = np.clip(np.nan_to_num(trans), 0.0, 1.0)

    lo, hi = np.percentile(a[finite], band) if finite.any() else (0.0, 1.0)
    ceiling = float(a[finite].max()) if finite.any() else np.inf
    # a soft ramp rather than a threshold: "loosely" is the point, a hard edge
    # here becomes a hard edge in the flow field two steps later
    w = np.clip((trans - open_at * 0.6) / (open_at * 0.6), 0.0, 1.0)
    w[a >= min(ceiling, hi * 2.5)] = 0.0                # clipped or flaring
    w[~finite] = 0.0
    return trans.astype(np.float32), w.astype(np.float32)


def temporal_transmission(stack: np.ndarray, blur: float, quantile: float = 85.0):
    """The fence's transmission, estimated from time instead of from viewpoint.

    The fence is the thing that does not move. The capsules sweep past behind
    it, so over a couple of dozen frames a high temporal quantile of each pixel
    is that pixel's transmission times the brightest thing that ever passed
    behind it - and the second factor is far smoother than the first, so
    dividing by a heavily blurred copy of the envelope leaves the fence.

    This is the same "see through it" operator the multi-view composite
    applies across viewpoints, run down the time axis instead, where there are
    twenty samples rather than four. It is done in the camera's own frame,
    before any warp, because that is where the fence sits.

    It only works across the motion. The capsules travel horizontally, so
    plenty of scene passes behind a VERTICAL bar and the envelope sees
    straight through it; behind a HORIZONTAL bar the same band of scene keeps
    coming round and no temporal statistic ever gets a look. Measured on
    burst0001, this takes 25-46% off the vertical family in all four views of
    the band and 0-4% off the horizontal one. The horizontal family is what
    the four viewpoints are for - their baselines run both ways.
    """
    # np.percentile sorts the whole 21-frame stack per pixel; a partition puts
    # only the wanted order statistic in place and is the difference between a
    # sequence taking twenty-five minutes and taking eight.
    if quantile >= 100:
        env = stack.max(axis=0)
    else:
        k = int(round(quantile / 100.0 * (stack.shape[0] - 1)))
        env = np.partition(stack, k, axis=0)[k]
    env = env.astype(np.float32)
    if cv2 is None:
        return np.ones_like(env)
    base = cv2.GaussianBlur(env, (0, 0), blur)
    return np.clip(env / np.maximum(base, 1e-6), 0.15, None)


def cross_view_transmission(aligned: dict, idxs, level_px: float):
    """The same see-through operator as the temporal one, run across viewpoints.

    The band's four cameras sit at the corners of a square, so BOTH baselines
    are present, and each one defeats the bar family it is perpendicular to:

        horizontal baseline (e00-e02)  ->  sees past the VERTICAL bars
        vertical baseline   (e00-e08)  ->  sees past the HORIZONTAL bars

    Measured on burst0001, the levelled maximum over one pair takes the family
    its baseline addresses down by about 95% and leaves the other; over all
    four it takes both. That is the whole answer to the horizontal family the
    temporal estimate cannot touch - the information was there all along.

    Two things make it work, and the pipeline was missing both.

    It has to be a MAXIMUM. A median is a consensus operator, and all four
    views agree that there is a fence somewhere, so the median leaves it in
    almost untouched: measured, the median of four scores 5.89 against a
    single view's 5.93.

    The views have to be put on a COMMON LEVEL first. e10 is brighter than the
    rest, so a raw maximum is mostly e10 and inherits e10's fence: 1.47
    levelled against 0.57 not.

    Returns {view: transmission}, each in (0, 1].
    """
    if cv2 is None:
        return {i: np.ones_like(np.nan_to_num(aligned[i])) for i in idxs}
    lev = {}
    for i in idxs:
        v = np.nan_to_num(aligned[i]).astype(np.float32)
        lev[i] = v / np.maximum(cv2.GaussianBlur(v, (0, 0), level_px), 1e-6)
    top = np.max(np.stack([lev[i] for i in idxs]), axis=0)
    return {i: np.clip(lev[i] / np.maximum(top, 1e-6), 0.15, 1.0) for i in idxs}


def bandpass(img: np.ndarray, small: float, large: float):
    """Keep the scales the target lives at; drop the rest.

    Below `small` is sensor noise and the speckle of a single glint. Above
    `large` is the illumination falloff and the fence's own smooth
    attenuation, which is exactly what must NOT drive a correlation - it is
    the thing being removed, and it is different in every view.
    """
    a = np.nan_to_num(img).astype(np.float32)
    if cv2 is None:
        return a - a.mean()
    return cv2.GaussianBlur(a, (0, 0), small) - cv2.GaussianBlur(a, (0, 0), large)


# ----------------------------------------------------------------- matching --
def block_match(ref: np.ndarray, mov: np.ndarray, w_ref: np.ndarray,
                w_mov: np.ndarray, block: int, stride: int, search: int,
                min_cover=0.25, min_peak=0.25):
    """Displacement of `mov` against `ref`, one vector per block.

    Masked normalised correlation, so the fence and the highlights take no
    part in deciding where a block belongs. Returns the grid of displacements,
    their peak scores, and a validity flag - a block is invalid when too
    little of it survived the mask, when the peak is weak, or when the winner
    sits on the edge of the search, which means the surface was still climbing
    and the number is where the window stopped rather than where the block is.
    """
    h, w = ref.shape
    ys = list(range(search, h - block - search + 1, stride))
    xs = list(range(search, w - block - search + 1, stride))
    fx = np.zeros((len(ys), len(xs)), np.float32)
    fy = np.zeros_like(fx)
    pk = np.zeros_like(fx)
    ok = np.zeros(fx.shape, bool)
    if cv2 is None or not ys or not xs:
        return fx, fy, pk, ok, ys, xs

    for gy, y in enumerate(ys):
        for gx, x in enumerate(xs):
            tpl = ref[y:y + block, x:x + block]
            m = w_ref[y:y + block, x:x + block]
            if m.mean() < min_cover:
                continue
            reg = mov[y - search:y + block + search, x - search:x + block + search]
            mw = w_mov[y - search:y + block + search, x - search:x + block + search]
            if mw.mean() < min_cover:
                continue
            try:
                surf = cv2.matchTemplate(reg * 1.0, tpl * m, cv2.TM_CCORR_NORMED,
                                         mask=m)
            except cv2.error:
                continue
            surf = np.nan_to_num(surf, nan=-1.0, posinf=-1.0, neginf=-1.0)
            r0, c0 = np.unravel_index(int(np.argmax(surf)), surf.shape)
            peak = float(surf[r0, c0])
            if peak < min_peak:
                continue
            if r0 in (0, surf.shape[0] - 1) or c0 in (0, surf.shape[1] - 1):
                continue                                # still climbing at the edge

            def sub(v):
                d = v[0] - 2 * v[1] + v[2]
                return float(np.clip((v[0] - v[2]) / (2 * d), -1, 1)) if d else 0.0

            dx = (c0 + sub(surf[r0, c0 - 1:c0 + 2])) - search
            dy = (r0 + sub(surf[r0 - 1:r0 + 2, c0])) - search
            fx[gy, gx], fy[gy, gx], pk[gy, gx], ok[gy, gx] = dx, dy, peak, True
    return fx, fy, pk, ok, ys, xs


def flow_is_real(a_fx, a_fy, a_ok, b_fx, b_fy, b_ok, ratio_at=1.5):
    """Is the flow a measurement, or just the search window's edge?

    A correlation surface with a true peak puts the winner in the same place
    however far it is allowed to look. A surface with no peak has its argmax
    wander outwards with the window, so the displacement grows in proportion
    to the search radius while the peak height does not improve. Running the
    match at two radii and comparing is the cheapest test of the difference,
    and it is worth the second run: a flow taken from a flat surface warps
    every block somewhere plausible-looking and wrong, and the damage is
    invisible afterwards.

    This catches a surface with no peak. It does NOT catch the other failure,
    which is a peak with nothing behind it - see the systematic-offset test
    where this is called.

    a is the half-radius run, b the full one. Returns (verdict, ratio).
    """
    if not a_ok.any() or not b_ok.any():
        return False, float("nan")
    da = float(np.median(np.hypot(a_fx[a_ok], a_fy[a_ok])))
    db = float(np.median(np.hypot(b_fx[b_ok], b_fy[b_ok])))
    if da <= 1e-6:
        return True, 0.0
    return (db / da) < ratio_at, db / da


def regularise_flow(fx, fy, ok, shape, ys, xs, block):
    """Grid of block displacements -> a dense, smooth flow field.

    Blocks that failed are filled from the ones that did not, the field is
    median filtered to throw out single wrong blocks, and then interpolated.
    A displacement field over a rigid scene should vary slowly; anything that
    does not is a mismatch, not a discovery.
    """
    if not ok.any():
        return (np.zeros(shape, np.float32), np.zeros(shape, np.float32))
    g = []
    for f in (fx, fy):
        v = f.copy()
        v[~ok] = np.nan
        # fill the gaps with the median of what is known, then smooth
        med = float(np.nanmedian(v))
        v = np.where(np.isfinite(v), v, med)
        if cv2 is not None and min(v.shape) >= 3:
            v = cv2.medianBlur(v.astype(np.float32), 3)
        g.append(v.astype(np.float32))
    h, w = shape
    out = []
    for v in g:
        if cv2 is None:
            out.append(np.full(shape, float(v.mean()), np.float32))
            continue
        # the grid samples sit at block centres, so place them there and
        # interpolate over the full frame
        big = cv2.resize(v, (len(xs), len(ys)), interpolation=cv2.INTER_NEAREST)
        cy = np.array(ys, np.float32) + block / 2
        cx = np.array(xs, np.float32) + block / 2
        yy = np.clip((np.arange(h) - cy[0]) / max(cy[-1] - cy[0], 1e-6), 0, 1) * (len(ys) - 1)
        xx = np.clip((np.arange(w) - cx[0]) / max(cx[-1] - cx[0], 1e-6), 0, 1) * (len(xs) - 1)
        mapy, mapx = np.meshgrid(yy, xx, indexing="ij")
        out.append(cv2.remap(big, mapx.astype(np.float32), mapy.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out[0], out[1]


def warp_by_flow(img: np.ndarray, fx: np.ndarray, fy: np.ndarray):
    """Move `img` by the flow, so that it lands on the reference."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return S.remap_image(img, (xx + fx).astype(np.float32),
                         (yy + fy).astype(np.float32))


# --------------------------------------------------------- decomposition ----
def separate(views: np.ndarray, blur: float, iters: int = 8, eps: float = 1e-3):
    """Split the aligned views into one background and a transmission each.

    Alternating least squares on I_i = T_i . B. Each half is the exact
    minimiser of the same residual with the other half held fixed:

        T_i = I_i / B                       (per view, per pixel)
        B   = sum_i T_i I_i / sum_i T_i^2   (the common background)

    and between the two the priors are applied, which is where the physics
    goes in:

    T is smoothed, but only just.  The fence is a tenth of the way to the
        target, so the lens leaves it soft and a transmission sharper than
        that would be fitting the background rather than the fence. The trap
        is the other way round: smoothing at a third of the fence period costs
        a Gaussian 90% of its response AT that period, so the fence can no
        longer be represented in T at all and lands in B instead - which is
        exactly the failure that leaves a fence in the "background". Keep it
        well under the period.
    T is normalised so that the clearest view of each pixel reads 1.  Without
        it the pair (T, B) is only determined up to a per-pixel scale and the
        whole thing drifts. The clearest view is taken as the second highest
        rather than the highest, so that one specular flash cannot define what
        "unoccluded" means.
    T is clipped to (0, 1].  A fence adds no light.

    Returns (B, T, history of the residual). It converges fast: on burst0001
    the result is identical at one iteration and at eight.
    """
    v = views.shape[0]
    I = np.nan_to_num(views).astype(np.float32)
    valid = np.isfinite(views).all(axis=0)
    srt = np.sort(I, axis=0)
    B = srt[max(0, v - 2)].copy()                       # second highest: the init
    hist = []
    for _ in range(iters):
        with np.errstate(invalid="ignore", divide="ignore"):
            T = I / np.maximum(B, eps)
        if cv2 is not None and blur > 0:
            T = np.stack([cv2.GaussianBlur(t, (0, 0), blur) for t in T])
        T = np.nan_to_num(T, nan=eps, posinf=1.0)
        ref = np.sort(T, axis=0)[max(0, v - 2)]         # the clearest view here
        with np.errstate(invalid="ignore", divide="ignore"):
            T = T / np.maximum(ref, eps)
        T = np.clip(np.nan_to_num(T), eps, 1.0)
        num = (T * I).sum(axis=0)
        den = (T * T).sum(axis=0)
        B = np.where(den > eps, num / np.maximum(den, eps), B)
        B = np.maximum(np.nan_to_num(B), 0.0)
        hist.append(float(np.sqrt(np.mean((I - T * B)[:, valid] ** 2))))
    return B.astype(np.float32), T.astype(np.float32), hist


def texture_fill(img: np.ndarray, hole: np.ndarray, patch: int, search: int):
    """Synthesise the holes from the texture around them.

    Where every view was behind a bar there is no measurement, and no amount
    of arithmetic will produce one. Diffusing the edges inward would give a
    smooth blob, which reads as an object that is not there - worse than a
    hole, because it looks like data. Copying real patches from nearby keeps
    the statistics of the surroundings and stays honestly synthetic.

    Greedy onion peel: the tile with the most known pixels is filled first,
    from the best-matching fully-known tile within a search window, compared
    over the pixels that ARE known. Filling the best-constrained tile first
    means each fill has the most evidence available to it.
    """
    out = np.nan_to_num(img).astype(np.float32).copy()
    todo = hole.copy()
    if cv2 is None or not todo.any():
        return out, todo
    h, w = out.shape
    half = patch // 2
    guard = 0
    while todo.any() and guard < 4000:
        guard += 1
        # the tile that needs filling and has the most context
        integral = cv2.boxFilter(todo.astype(np.float32), -1, (patch, patch),
                                 normalize=True)
        cand = (integral > 0) & (integral < 1.0)
        if not cand.any():
            cand = integral > 0
            if not cand.any():
                break
        score = np.where(cand, 1.0 - integral, -1.0)
        score[:half, :] = -1; score[-half:, :] = -1
        score[:, :half] = -1; score[:, -half:] = -1
        if score.max() <= 0:
            break
        cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
        y0, x0 = cy - half, cx - half
        tile = out[y0:y0 + patch, x0:x0 + patch]
        known = ~todo[y0:y0 + patch, x0:x0 + patch]
        if known.mean() < 0.05:
            todo[y0:y0 + patch, x0:x0 + patch] = False   # nothing to go on
            continue
        sy0, sx0 = max(0, y0 - search), max(0, x0 - search)
        sy1 = min(h, y0 + patch + search); sx1 = min(w, x0 + patch + search)
        reg = out[sy0:sy1, sx0:sx1]
        regh = todo[sy0:sy1, sx0:sx1].astype(np.float32)
        if reg.shape[0] <= patch or reg.shape[1] <= patch:
            todo[y0:y0 + patch, x0:x0 + patch] = False
            continue
        m = known.astype(np.float32)
        res = cv2.matchTemplate(reg, tile * m, cv2.TM_CCORR_NORMED, mask=m)
        # never take a source that is itself a hole
        holey = cv2.matchTemplate(regh, np.ones_like(tile), cv2.TM_CCORR)
        res = np.where(holey > 0.02 * patch * patch, -1.0, np.nan_to_num(res, nan=-1.0))
        by, bx = np.unravel_index(int(np.argmax(res)), res.shape)
        src = reg[by:by + patch, bx:bx + patch]
        fill = todo[y0:y0 + patch, x0:x0 + patch]
        tile[fill] = src[fill]
        todo[y0:y0 + patch, x0:x0 + patch] = False
    return out, hole


# ------------------------------------------------------------------- driver --
def _fence_spectrum(a: np.ndarray, axis: int):
    v = np.nan_to_num(a)[a.shape[0] // 6:-a.shape[0] // 6,
                         a.shape[1] // 6:-a.shape[1] // 6].astype(np.float64)
    if axis == 0:
        v = v.T
    return np.abs(np.fft.rfft(v - v.mean(axis=1, keepdims=True), axis=1)).mean(0), v.shape[1]


def fence_period(a: np.ndarray, axis: int, lo=40.0, hi=110.0):
    """The period of one bar family: axis 1 for vertical bars, 0 for horizontal.

    The two families are not at the same period and not at the same period in
    every view - a woven mesh on a rotating cylinder is not one lattice - so
    each is measured on its own.
    """
    sp, w = _fence_spectrum(a, axis)
    ks = [k for k in range(6, len(sp)) if lo <= w / k <= hi]
    return (w / max(ks, key=lambda k: sp[k])) if ks else float("nan")


def fence_excess(a: np.ndarray, axis: int, period: float, lo=40.0, hi=110.0):
    """The PERIODIC part at the fence line, as a percent of the frame's energy.

    Not the share of the three bins around the peak: those bins also hold
    whatever broadband scene lives at that frequency, and on the horizontal
    family the fence peak stands only 1.2-1.6x above that floor, so a share is
    mostly reporting on the scene and barely moves however well the fence is
    removed. Subtracting the band's own median leaves the part that is
    actually periodic, which is the part the fence is responsible for.
    """
    sp, w = _fence_spectrum(a, axis)
    k = int(round(w / period))
    if not 3 <= k < len(sp) - 2:
        return float("nan")
    band = [j for j in range(6, len(sp)) if lo <= w / j <= hi]
    floor = float(np.median([sp[j] for j in band])) if band else 0.0
    return max(float(sp[k - 1:k + 2].sum()) - 3 * floor, 0.0) / \
        float(sp[3:len(sp) // 3].sum()) * 100


def fence_share(a: np.ndarray, period: float):
    """Kept for the sweep csv: the raw share, not the excess."""
    sp, w = _fence_spectrum(a, 1)
    k = int(round(w / period))
    if not 3 <= k < len(sp) - 2:
        return float("nan")
    return float(sp[k - 1:k + 2].sum() / sp[3:len(sp) // 3].sum()) * 100


def detail(a: np.ndarray):
    """Laplacian energy - how much structure survived, at all."""
    v = np.ascontiguousarray(np.nan_to_num(a).astype(np.float32))
    return float(np.mean(cv2.Laplacian(v, cv2.CV_32F) ** 2)) if cv2 is not None else 0.0


def defence_at(raw, band_cams, ref, z, calib, period, target_px, f_px, args,
               out=None, verbose=True, fill=True, metric=None):
    """Every stage at one plane. -> dict of images and numbers.

    `out` writes the per-stage images; a sweep passes None and keeps only the
    result, because seventeen planes of six stages is a hundred images nobody
    will look at.

    `metric` is (view, vertical period, horizontal period) taken from the views
    AS CAPTURED. Both halves have to come from outside: the transmissions
    rewrite the views in place, so a reference read from them is already
    corrected and the improvement reads as a fraction of itself - and a period
    measured on them lands on whatever survived, which on 750 and 900 nm was
    104 px, a frequency the fence is not at.
    """
    say = print if verbose else (lambda *a, **k: None)
    warped = {i: S.warp_for_plane(raw[i], i, ref, z, f_px, args.pitch, calib)[0]
              for i in band_cams}

    before = np.array(warped[ref], copy=True)     # for the comparison panels

    # ---- 1b the cross-view transmission ------------------------------------
    # Before the matching, not after. The fence is the loudest thing in the
    # frame and it is in a DIFFERENT PLACE in each view, so while it is there
    # it does not merely fail to help the correlation - it actively drags the
    # winner to the edge of the search window. Measured on burst0001: with the
    # fence in, the block match reports a displacement of about 0.8 times
    # whatever radius it is given, at a peak of 0.4-0.6, on a quarter of the
    # blocks. With it out, the same match returns 1.3 px at every radius from
    # +-2 to +-24, at a peak of 0.91-0.95, on three quarters of the blocks.
    # The residual the calibration leaves really is about a pixel; it was
    # never visible underneath the fence.
    if args.cross_view > 0:
        xv = cross_view_transmission(warped, band_cams, args.cross_view)
        for i in band_cams:
            warped[i] = np.nan_to_num(warped[i]) / np.maximum(xv[i], 0.15)
        say(f"cross   : transmission from the levelled max of {len(band_cams)} views, "
            f"level {args.cross_view:.0f}px")
        if out:
            for i in band_cams:
                S.write_png(out / f"1_crossview_{S.cam_name(i)}.png",
                            S.to_u8(xv[i], lo=0, hi=1))

    # ---- 2 mask ------------------------------------------------------------
    weight = {}
    for i in band_cams:
        weight[i] = view_masks(warped[i], period)[1]
        if out:
            S.write_png(out / f"2_mask_{S.cam_name(i)}.png", S.to_u8(weight[i], lo=0, hi=1))
    say("mask    : matchable area per view "
        + ", ".join(f"{S.cam_name(i)} {weight[i].mean()*100:.0f}%" for i in band_cams))

    # ---- 3 match, 4 warp locally -------------------------------------------
    small, large = max(1.0, target_px / 8), max(2.0, target_px * 1.2)
    bp = {i: bandpass(warped[i], small, large) for i in band_cams}
    aligned, n_bad = {ref: warped[ref]}, 0
    for i in band_cams:
        if i == ref:
            continue
        t0 = time.time()
        half = block_match(bp[ref], bp[i], weight[ref], weight[i],
                           args.block, args.block // 2, max(4, args.search // 2))
        fx, fy, pk, ok, ys, xs = block_match(
            bp[ref], bp[i], weight[ref], weight[i],
            args.block, args.block // 2, args.search)
        n_ok = int(ok.sum())
        real, ratio = flow_is_real(half[0], half[1], half[3], fx, fy, ok)
        d_half = float(np.median(np.hypot(half[0][half[3]], half[1][half[3]]))) \
            if half[3].any() else float("nan")
        d_full = float(np.median(np.hypot(fx[ok], fy[ok]))) if ok.any() else float("nan")
        peak = float(np.median(pk[ok])) if ok.any() else float("nan")
        # The median MAGNITUDE cannot be zero: for displacements scattered
        # about zero with a spread of one pixel it reads about 1.2 whatever
        # the truth is. What says whether there is anything to correct is the
        # systematic part - the vector mean. On burst0001 that is 0.03-0.4 px
        # against a scatter of 0.9-1.9, so the calibrated warp leaves no
        # measurable residual, and applying the scatter as a warp costs 40% of
        # the detail while gaining nothing. Measure it, report it, warp only
        # when there is something worth warping for.
        mdx = float(np.mean(fx[ok])) if ok.any() else 0.0
        mdy = float(np.mean(fy[ok])) if ok.any() else 0.0
        systematic = float(np.hypot(mdx, mdy))
        note = ""
        if not real and not args.force_flow:
            fx, fy, ok = np.zeros_like(fx), np.zeros_like(fy), ok & False
            note = (f"  -- |d| grew {ratio:.1f}x when the search doubled, so this is "
                    f"the window's edge, not a match: global alignment kept")
            n_bad += 1
        elif systematic < args.flow_min and not args.force_flow:
            fx, fy, ok = np.zeros_like(fx), np.zeros_like(fy), ok & False
            note = (f"  -- systematic offset {systematic:.2f}px is under "
                    f"{args.flow_min:.2f}, so there is nothing to correct: "
                    f"global alignment kept")
            n_bad += 1
        FX, FY = regularise_flow(fx, fy, ok, warped[ref].shape, ys, xs, args.block)
        aligned[i] = warp_by_flow(warped[i], FX, FY) if ok.any() else warped[i]
        say(f"match   : {S.cam_name(i)}  {n_ok}/{ok.size} blocks  "
            f"scatter {d_half:.1f}->{d_full:.1f}px  systematic {systematic:.2f}px  "
            f"peak {peak:.2f}  {time.time()-t0:.1f}s{note}")
        if out:
            S.write_png(out / f"3_flow_{S.cam_name(i)}.png",
                        S.to_u8(np.hypot(FX, FY), lo=0, hi=max(1.0, args.search)))
    if n_bad and verbose:
        say(f"match   : {n_bad} of {len(band_cams)-1} view(s) left on the global "
            f"alignment. --force-flow warps by the flow regardless")
    stack = np.stack([aligned[i] for i in band_cams])

    # ---- 5 separate --------------------------------------------------------
    B, T, hist = separate(stack, blur=max(1.0, period * args.t_blur), iters=args.iters)
    say(f"separate: residual {hist[0]:.2f} -> {hist[-1]:.2f} over {len(hist)} iterations")
    if out:
        for k, i in enumerate(band_cams):
            S.write_png(out / f"5_transmission_{S.cam_name(i)}.png",
                        S.to_u8(T[k], lo=0, hi=1))
        S.write_png(out / "5_background.png", S.to_u8(B))

    # ---- 6 fill ------------------------------------------------------------
    hole = T.max(axis=0) < args.hole_at
    filled = B
    if hole.any() and not fill:
        # A sweep scores B, not the synthesis, and the greedy fill is the most
        # expensive thing here - seventeen planes of it is minutes for numbers
        # nobody reads.
        say(f"fill    : {hole.mean()*100:.2f}% would be synthesised (skipped in a sweep)")
    elif hole.any():
        t0 = time.time()
        filled, _ = texture_fill(B, hole, int(max(8, target_px // 2)),
                                 int(max(16, target_px * 3)))
        say(f"fill    : {hole.mean()*100:.2f}% of the frame had no clear view in any "
            f"camera, synthesised in {time.time()-t0:.1f}s")
    else:
        say("fill    : every pixel had a clear view somewhere; nothing to synthesise")

    med = S.robust_stats(stack)[0]
    m_ref, pv, ph = (metric if metric else
                     (before, fence_period(before, 1), fence_period(before, 0)))
    fam = {n: {"v": fence_excess(im, 1, pv), "h": fence_excess(im, 0, ph)}
           for n, im in (("reference", m_ref), ("median", med), ("background", B))}
    res = {"z": z, "B": B, "T": T, "filled": filled, "hole": hole, "median": med,
           "before": before, "periods": (pv, ph), "family": fam,
           "reference": warped[ref], "residual": hist, "flows_rejected": n_bad,
           "fence_ref": fence_share(warped[ref], period),
           "fence_median": fence_share(med, period),
           "fence_B": fence_share(B, period),
           "fence_filled": fence_share(filled, period),
           "detail_B": detail(B), "detail_median": detail(med),
           "hole_frac": float(hole.mean())}
    say(f"fence   : periodic excess at the fence line, {pv:.0f}px vertical / "
        f"{ph:.0f}px horizontal bars")
    for n in ("reference", "median", "background"):
        say(f"          {n:<10} vertical {fam[n]['v']:5.2f}%   "
            f"horizontal {fam[n]['h']:5.2f}%")
    if out:
        S.write_png(out / "6_hole.png", S.to_u8(hole.astype(np.float32), lo=0, hi=1))
        S.write_png(out / "6_defenced.png",
                    S.tone_map(S.to_u8(filled), args.gamma, args.clahe))
        S.write_png(out / "0_reference_view.png",
                    S.tone_map(S.to_u8(warped[ref]), args.gamma, args.clahe))
        S.write_png(out / "0_median.png",
                    S.tone_map(S.to_u8(med), args.gamma, args.clahe))
    return res


class FrameWindow:
    """The frames a sliding temporal window needs, loaded once each.

    A 21-frame window recomputed per output frame would read 21 frames per
    camera per frame - for a 90-frame burst that is 7560 reads of a 178 MB
    archive. Walking the sequence in order and keeping a ring buffer makes it
    one read per camera per frame, plus the window to prime it.
    """

    def __init__(self, load, cams, n):
        self.load, self.cams, self.n = load, cams, max(1, n)
        self.buf = {i: {} for i in cams}

    def at(self, i, k, n_frames):
        """The window centred on k for camera i, clamped to the burst."""
        lo = max(0, min(k - self.n // 2, n_frames - self.n))
        want = range(lo, min(lo + self.n, n_frames))
        b = self.buf[i]
        for j in want:
            if j not in b:
                b[j] = self.load(i, j)
        for j in [j for j in b if j not in want]:
            del b[j]
        return np.stack([b[j] for j in want])


def run(args):
    label, cams = S.discover_burst(args.burst)
    band_cams = [i for i in sorted(cams) if S.cam_band(i) == args.band]
    if len(band_cams) < 2:
        raise SystemExit(f"{args.band}nm has {len(band_cams)} view(s) in this burst")
    calib = S.load_calibration(args.calib) if args.calib else None
    offsets = {}
    if args.offsets:
        opath = (Path("sa_results") / args.burst.parent.name /
                 f"temporal_align_b{int(''.join(c for c in label if c.isdigit()) or 0):04d}.json"
                 if args.offsets == "auto" else Path(args.offsets))
        offsets = S.load_offsets(opath, band_cams)[0] if opath.exists() else {}

    # the band belongs in the path: four of them into one directory is four
    # runs overwriting each other
    out = (args.out or Path.cwd() / "output" / "defence" /
           args.burst.parent.name / label / f"{args.band}nm")
    out.mkdir(parents=True, exist_ok=True)
    print(f"burst   : {label}   band {args.band}nm   views "
          f"{', '.join(S.cam_name(i) for i in band_cams)}")

    def one_frame(i, k):
        p, pre = cams[i]
        im, _, _ = S.load_camera_frame(p, pre, k)
        if args.scale != 1.0:
            im = cv2.resize(im, None, fx=args.scale, fy=args.scale,
                            interpolation=cv2.INTER_AREA)
        return im

    probe_path, probe_pre = cams[band_cams[0]]
    n_frames = S.load_camera_frame(probe_path, probe_pre, 0)[1]
    spec = args.frames if args.frames else str(args.frame)
    seq = S.parse_frames(spec, n_frames)
    if not seq:
        raise SystemExit(f"no frame selected out of {n_frames}")
    args.frame = seq[len(seq) // 2]            # the probe decides the settings
    window = FrameWindow(one_frame, band_cams, args.temporal)

    def views_at(idx):
        """One instant, gain-matched and with its fence transmission removed."""
        v = {i: one_frame(i, idx + offsets.get(i, 0)) for i in band_cams}
        v, _ = S.band_gain_normalise(v, {args.band: band_cams})
        if args.temporal > 1:
            for i in band_cams:
                st = window.at(i, idx + offsets.get(i, 0), n_frames)
                t = temporal_transmission(st, blur=period * 1.3,
                                          quantile=args.temporal_q)
                v[i] = v[i] / t
        return v

    raw = {}
    for i in band_cams:
        raw[i] = one_frame(i, args.frame + offsets.get(i, 0))
    raw, _ = S.band_gain_normalise(raw, {args.band: band_cams})
    plain_views = {i: np.array(raw[i], copy=True) for i in band_cams}

    # The fence period is measured HERE, on the views as captured. After the
    # temporal correction the fence line is most of the way gone, which is the
    # point - but it also means the period can no longer be read off the data,
    # and every window below is sized in fence periods.
    est = [(S.fence_period_px(raw[i])[1], S.fence_period_px(raw[i])[0]) for i in band_cams]
    period = float(max(est)[1])
    per_share = float(max(est)[0])

    if args.temporal > 1:
        print(f"temporal: {args.temporal} frames, {args.temporal_q:.0f}th percentile, "
              f"blur {period*1.3:.0f}px -- fence transmission taken out per view")
        raw = views_at(args.frame)
    f_px = args.fpx * args.scale

    clear = {i: view_masks(raw[i], period)[1].mean() for i in band_cams}
    ref = args.ref if args.ref in band_cams else max(clear, key=clear.get)
    print(f"ref     : {S.cam_name(ref)}, the clearest view "
          + ", ".join(f"{S.cam_name(i)} {clear[i]*100:.0f}%" for i in band_cams))
    f_ref = (calib["cams"][ref]["K"][0, 0] * args.scale
             if calib and ref in calib["cams"] else f_px)
    target_px = args.target_size_mm * f_ref / args.z
    # the yardstick: the reference view warped to the plane and otherwise
    # untouched, plus the two bar periods measured on it
    plain_ref = S.warp_for_plane(plain_views[ref], ref, ref, args.z, f_px,
                                 args.pitch, calib)[0]
    metric = (plain_ref, fence_period(plain_ref, 1), fence_period(plain_ref, 0))
    print(f"scene   : fence period {period:.1f}px ({per_share*100:.0f}% of the band "
          f"as captured), target {target_px:.0f}px at {args.z:.0f}mm")

    if len(seq) > 1:
        # ---- the sequence ---------------------------------------------------
        # One 8-bit mapping for the whole run, taken from the probe frame:
        # recomputing it per frame ties the display to the scene's own
        # brightness and the video flickers, which is the thing a de-fenced
        # sequence exists to let you judge.
        print(f"frames  : {len(seq)} of {n_frames} ({seq[0]}..{seq[-1]})")
        probe = defence_at(raw, band_cams, ref, args.z, calib, period, target_px,
                           f_px, args, out=out, verbose=True, metric=metric)

        def plain_at(idx):
            """The same instant with no transmissions taken out at all - what a
            camera and a plain median actually give, which is what the result
            has to be compared against."""
            v = {i: one_frame(i, idx + offsets.get(i, 0)) for i in band_cams}
            v, _ = S.band_gain_normalise(v, {args.band: band_cams})
            w = {i: S.warp_for_plane(v[i], i, ref, args.z, f_px, args.pitch, calib)[0]
                 for i in band_cams}
            return w[ref], S.robust_stats(np.stack([w[i] for i in band_cams]))[0]

        # One 8-bit range for the whole run, pooled over frames spread across it:
        # the drum's illumination sweeps, so a range taken from one frame leaves
        # a third of the sequence nearly black.
        rng = {}
        sample = [seq[k] for k in np.linspace(0, len(seq) - 1, min(6, len(seq))).astype(int)]
        pools = {"defenced": [], "reference": [], "median": []}
        for idx in sample:
            r = (probe if idx == args.frame else
                 defence_at(views_at(idx), band_cams, ref, args.z, calib, period,
                            target_px, f_px, args, out=None, verbose=False))
            pr, pm = plain_at(idx)
            pools["defenced"].append(r["filled"])
            pools["reference"].append(pr)
            pools["median"].append(pm)
        for name, ims in pools.items():
            v = np.concatenate([np.nan_to_num(im).ravel()[::17] for im in ims])
            rng[name] = tuple(float(x) for x in np.percentile(v, [1.0, 99.0]))
        counts, t0 = {}, time.time()
        for n, idx in enumerate(seq, 1):
            r = (probe if idx == args.frame else
                 defence_at(views_at(idx), band_cams, ref, args.z, calib, period,
                            target_px, f_px, args, out=None, verbose=False))
            pr, pm = plain_at(idx)
            tiles = {}
            for name, im in (("defenced", r["filled"]), ("median", pm),
                             ("reference", pr)):
                lo, hi = rng[name]
                u8 = S.tone_map(S.to_u8(im, lo=lo, hi=hi), args.gamma, args.clahe)
                d = out / "png" / name
                d.mkdir(parents=True, exist_ok=True)
                S.write_png(d / f"frame{idx:04d}.png", u8)
                counts[name] = counts.get(name, 0) + 1
                tiles[name] = u8
                if args.save_float:
                    # the values before any 8-bit range, so that panels can be shown
                    # on one shared scale later (the strips below are not)
                    fd = out / "float"
                    fd.mkdir(parents=True, exist_ok=True)
                    np.save(fd / f"{name}_frame{idx:04d}.npy", np.asarray(im, np.float32))
            # the three side by side, which is the comparison worth watching
            side = np.hstack([S.label_tile(tiles[k], t) for k, t in
                              (("reference", "one camera"), ("median", "median of four"),
                               ("defenced", "de-fenced"))])
            d = out / "png" / "compare"
            d.mkdir(parents=True, exist_ok=True)
            S.write_png(d / f"frame{idx:04d}.png", side)
            counts["compare"] = counts.get("compare", 0) + 1
            if n == 1 or n % 10 == 0 or n == len(seq):
                el = time.time() - t0
                print(f"  [{n}/{len(seq)}] frame {idx:04d}   {el/n:.2f}s/frame   "
                      f"eta {S.human_time((len(seq)-n) * el / n)}")
        vids = []
        if not args.no_video:
            (out / "video").mkdir(exist_ok=True)
            for name in sorted(counts):
                v = S.encode_video(out / "png" / name, out / "video" / f"{name}.mp4",
                                   args.fps)
                if v:
                    vids.append(v.name)
            print(f"videos  : {', '.join(vids) if vids else '(none written)'}")
        best = probe
    elif not args.sweep:
        res = defence_at(raw, band_cams, ref, args.z, calib, period, target_px,
                         f_px, args, out=out, metric=metric)
        best = res
    else:
        # Uniform in disparity, not in millimetres, and over the range the
        # target actually occupies - the same rule sa_packed uses, for the
        # same reason: at 400-700 mm a 100 mm step moves the aperture's
        # extremes by tens of pixels, at 3000 mm by three.
        sc = S.scene_scales(f_ref, args.occluder_mm, args.z, args.target_size_mm,
                            args.pitch)
        planes = S.sweep_planes(args.sweep, sc, f_ref, args.pitch, 1.0)
        sd = out / "sweep"; sd.mkdir(exist_ok=True)
        print(f"sweep   : {len(planes)} planes {planes[0]:.0f}..{planes[-1]:.0f}mm, "
              f"uniform in disparity")
        print(f"  {'z_mm':>7} {'fence B':>9} {'fence med':>10} {'detail B':>10} "
              f"{'hole%':>7} {'phase':>6} {'flows':>6}")
        rows, best = [], None
        for z in planes:
            tp = args.target_size_mm * f_ref / z
            r = defence_at(raw, band_cams, ref, z, calib, period, tp, f_px, args,
                           out=None, verbose=False, fill=False)
            ph = S.fence_phase_spread(band_cams, ref, z, f_ref, args.pitch,
                                      args.occluder_mm, period)
            r["phase"] = ph
            rows.append(r)
            print(f"  {z:7.1f} {r['fence_B']:8.2f}% {r['fence_median']:9.2f}% "
                  f"{r['detail_B']:10.1f} {r['hole_frac']*100:6.2f}% {ph:6.2f} "
                  f"{len(band_cams)-1-r['flows_rejected']:2d}/{len(band_cams)-1}")
            S.write_png(sd / f"defenced_z{int(round(z)):05d}.png",
                        S.tone_map(S.to_u8(r["filled"]), args.gamma, args.clahe))
            if best is None or r["fence_B"] < best["fence_B"]:
                best = r
        (out / "sweep.csv").write_text(
            "z_mm,fence_background_pct,fence_median_pct,detail_background,"
            "hole_fraction,fence_phase,flows_kept,"
            "excess_ref_v,excess_ref_h,excess_med_v,excess_med_h,"
            "excess_bg_v,excess_bg_h\n"
            + "".join(f"{r['z']:.1f},{r['fence_B']:.3f},{r['fence_median']:.3f},"
                      f"{r['detail_B']:.1f},{r['hole_frac']:.4f},{r.get('phase', float('nan')):.3f},"
                      f"{len(band_cams)-1-r['flows_rejected']},"
                      f"{r['family']['reference']['v']:.3f},{r['family']['reference']['h']:.3f},"
                      f"{r['family']['median']['v']:.3f},{r['family']['median']['h']:.3f},"
                      f"{r['family']['background']['v']:.3f},{r['family']['background']['h']:.3f}\n"
                      for r in rows))
        print(f"plane   : least fence at z = {best['z']:.0f} mm "
              f"({best['fence_B']:.2f}%, median there {best['fence_median']:.2f}%)")
        if cv2 is not None and len(rows) >= 4:
            pick = [rows[k] for k in np.linspace(0, len(rows) - 1, 4).astype(int)]
            S.write_png(out / "sweep_montage.png", S.mosaic_2x2(
                [S.label_tile(S.tone_map(S.to_u8(r["filled"]), args.gamma, args.clahe),
                              f"{r['z']:.0f}mm  fence {r['fence_B']:.2f}%")
                 for r in pick]))
        # and the full stage set at the winner, so it can be inspected
        defence_at(raw, band_cams, ref, best["z"], calib, period,
                   args.target_size_mm * f_ref / best["z"], f_px, args, out=out)

    (out / "defence.json").write_text(json.dumps({
        "burst": label, "band": args.band, "views": [S.cam_name(i) for i in band_cams],
        "reference": S.cam_name(ref), "frame": args.frame, "z_mm": best["z"],
        "scale": args.scale, "fence_period_px": period, "block": args.block,
        "search": args.search, "hole_fraction": best["hole_frac"],
        "fence_pct": {"reference": best["fence_ref"], "median": best["fence_median"],
                      "background": best["fence_B"], "filled": best["fence_filled"]},
        "frame_offsets": {S.cam_name(i): offsets[i] for i in sorted(offsets)},
    }, indent=2))
    print(f"wrote   : {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("burst", type=Path)
    ap.add_argument("--band", type=int, default=800, choices=S.BANDS)
    ap.add_argument("--z", type=float, default=550.0, help="focus plane, mm")
    ap.add_argument("--frame", type=int, default=45)
    ap.add_argument("--frames", default=None,
                    help="a sequence instead of one frame: all | 10-70 | 10:70:5. "
                         "Writes a png per frame per layer and an mp4 of each, plus a "
                         "'compare' strip of one camera, the median and the result")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--save-float", action="store_true",
                    help="with --frames: also save each panel as float .npy under float/. The "
                         "png panels each get their own 8-bit range, so they are not comparable "
                         "in brightness; these are")
    ap.add_argument("--no-video", action="store_true",
                    help="write the png sequences only")
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--offsets", nargs="?", const="auto", default=None)
    ap.add_argument("--ref", type=int, default=-1, help="reference camera index")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pitch", type=float, default=40.0)
    ap.add_argument("--fpx", type=float, default=6.0e-3 / 3.45e-6)
    ap.add_argument("--target-size-mm", type=float, default=20.0)
    ap.add_argument("--block", type=int, default=96,
                    help="matching block, px (default 96 - about one target)")
    ap.add_argument("--search", type=int, default=8,
                    help="how far a block may move, px. This is the residual the "
                         "calibration left, not the parallax, and once the fence is out "
                         "of the way it measures about 1.3 px - so keep it small. A wide "
                         "window on a surface with no peak just wanders to its edge")
    ap.add_argument("--sweep", default=None, nargs="?", const="auto",
                    help="run every plane instead of one: start:stop[:step] in mm, or "
                         "bare --sweep for the range the target occupies. Writes the "
                         "result at each, a csv and a montage")
    ap.add_argument("--occluder-mm", type=float, default=100.0,
                    help="lens front to the fence, mm - used for the fence-phase "
                         "column, which says whether a plane can cancel it at all")
    ap.add_argument("--cross-view", type=float, default=0.0,
                    help="estimate each view's fence transmission from the levelled "
                         "maximum across the views and divide it out (65 is a good "
                         "value, 0 off). Works on both bar families, because both "
                         "baselines are present in the 2x2")
    ap.add_argument("--temporal", type=int, default=0,
                    help="estimate each view's fence transmission from this many frames "
                         "and divide it out before warping (21 is a good value, 0 off). "
                         "Works on the bar family across the scene's motion")
    ap.add_argument("--temporal-q", type=float, default=85.0,
                    help="percentile for the temporal envelope (default 85)")
    ap.add_argument("--flow-min", type=float, default=0.5,
                    help="the systematic offset a view must show before the local flow "
                         "is applied, px (default 0.5). Below it the flow is scatter "
                         "about zero, and warping by it only costs sharpness")
    ap.add_argument("--force-flow", action="store_true",
                    help="warp by the local flow even when it fails the two-radius "
                         "test. For looking at what it would have done")
    ap.add_argument("--iters", type=int, default=3,
                    help="alternating least squares iterations (default 3). It converges "
                         "in one on burst0001 - the fence excess is identical at 1 and at "
                         "8 - so this is margin, not need")
    ap.add_argument("--t-blur", type=float, default=0.12,
                    help="transmission smoothing, in fence periods (default 0.12). "
                         "Anything approaching 0.3 removes the fence from T and puts "
                         "it back in the background")
    ap.add_argument("--hole-at", type=float, default=0.35,
                    help="below this transmission in every view, the pixel was "
                         "never seen and gets synthesised")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--clahe", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
