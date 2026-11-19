"""What moves, once what stands still is taken out.

The fence is 96 mm from the lens and does not move. The drum and the capsules
inside it do. So the median of a short run of frames is the fence - whatever
held a pixel for most of the window - and subtracting it leaves the scene.

This matters for a clear capsule specifically. It has no edge to threshold:
it refracts the background rather than blocking it, so brightness-based
detection finds its specular core and calls that the object. Motion does not
care about any of that. Whatever the capsule does to the light behind it, it
does somewhere else a frame later, and the difference marks the whole shape.

    python out/measure/motion.py burst0300 1 85              # one frame
    python out/measure/motion.py burst0300 1 85 win=15 z=700 mm=10,15
    python out/measure/motion.py burst0300 1 40-89 video=1   # a sequence

Sizes are quoted for a target of `mm` at `z`; the defaults are this session's
teardrops, 10-15 mm at about 700 mm.
"""
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, "analyze")
import sa_packed as S                                            # noqa: E402

opts = {}
pos = []
for a in sys.argv[1:]:
    (opts.__setitem__(*a.split("=", 1)) if "=" in a else pos.append(a))
BURST = pos[0] if pos else "burst0300"
CAM = int(pos[1]) if len(pos) > 1 else 1
FRAMES = pos[2] if len(pos) > 2 else "85"
WIN = int(opts.get("win", 15))              # frames in the median background
Z = float(opts.get("z", 700.0))
MM = [float(x) for x in opts.get("mm", "10,15").split(",")]
K = float(opts.get("k", 6.0))               # detection threshold, in noise sigma
NOTCH = opts.get("notch", "1") not in ("0", "false", "no")
VIDEO = opts.get("video", "0") not in ("0", "false", "no")
OUT = Path(opts.get("out", "out/measure"))
PACKED = Path("collected_data/packed/20260728_073439")

OUT.mkdir(parents=True, exist_ok=True)
_, cams = S.discover_burst(PACKED / BURST)
calib = S.load_calibration("calibration_results.npz")
f_px = calib["cams"][CAM]["K"][0, 0]
p, pre = cams[CAM]
_, n_frames, _ = S.load_camera_frame(p, pre, 0)

if "-" in FRAMES:
    a, b = FRAMES.split("-")
    targets = list(range(int(a), int(b) + 1))
else:
    targets = [int(x) for x in FRAMES.split(",")]

_probe, _, _ = S.load_camera_frame(p, pre, targets[0])
mx, my = S.sample_map(calib, CAM, CAM, Z, _probe.shape)

_cache = {}
def view(i):
    """One undistorted frame, levelled so a flicker in exposure is not motion."""
    i = max(0, min(n_frames - 1, i))
    if i not in _cache:
        im, _, _ = S.load_camera_frame(p, pre, i)
        v = np.nan_to_num(S.remap_image(im, mx, my)).astype(np.float32)
        med = float(np.median(v))
        _cache[i] = v * (12.0 / med) if med > 1 else v
        if len(_cache) > 3 * WIN:
            _cache.pop(next(iter(_cache)))
    return _cache[i]


def background(centre):
    """The median of the window: what held still while the drum turned."""
    lo = max(0, min(n_frames - WIN, centre - WIN // 2))
    return np.median(np.stack([view(i) for i in range(lo, lo + WIN)]), axis=0)


px_lo, px_hi = (m * f_px / Z for m in (min(MM), max(MM)))
print(f"{BURST} {S.cam_name(CAM)}: {len(targets)} frame(s), window {WIN}, "
      f"plane {Z:.0f} mm, f {f_px:.0f} px")
print(f"  a {min(MM):g}-{max(MM):g} mm object at {Z:.0f} mm spans "
      f"{px_lo:.0f}-{px_hi:.0f} px")

rows, all_hits = [], 0
def local_contrast(a, k=9):
    a = a.astype(np.float32)
    mu = cv2.blur(a, (k, k))
    mu2 = cv2.blur(a * a, (k, k))
    return np.sqrt(np.maximum(mu2 - mu * mu, 0)) / np.maximum(mu, 1.0)


period = float("nan")
if NOTCH:
    # measured on the residual, not the frame: the frame's strongest period is
    # the static fence, which the median already took out, while the lattice
    # left in the residual is the drum's own mesh turning with it
    period, _ = S.fence_period_px(view(targets[0]) - background(targets[0]))
    print(f"  lattice period {period:.1f} px -> notching it out of the residual"
          if np.isfinite(period) else "  no periodic lattice found; not notching")

for t in targets:
    cur, bg = view(t), background(t)
    res = cur - bg
    # The drum's own mesh moves with it, so it survives the median and fills
    # the residual with a lattice that swamps everything compact. It is
    # periodic and a capsule is not, which is the one difference a notch can
    # act on.
    if NOTCH and np.isfinite(period):
        res = S.notch_fence(res, period)
    # noise scale from the residual itself: most of the frame is fence that
    # cancelled, so its spread is what "no motion" looks like here
    sigma = 1.4826 * np.median(np.abs(res - np.median(res)))
    m = (np.abs(res) > K * sigma).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    lc_cur = local_contrast(cur)
    hits = []
    for k in range(1, n):
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        long_side = max(w, h)
        if not (px_lo * 0.7 <= long_side <= px_hi * 1.6):
            continue
        if max(w, h) / max(min(w, h), 1) > 3.0:          # not a streak of fence
            continue
        area = st[k, cv2.CC_STAT_AREA]
        if area / float(w * h) < 0.35:
            continue
        vals = res[lab == k]
        # transparent body or specular glint? A clear capsule refracts what is
        # behind it: the texture inside its outline is displaced and softened,
        # so local contrast DROPS. A glint adds light: contrast rises. This is
        # the one test that separates the two without an edge to measure.
        cx_, cy_ = int(cen[k][0]), int(cen[k][1])
        r = int(max(w, h))
        bx0, by0 = max(0, cx_ - w // 2), max(0, cy_ - h // 2)
        bx1, by1 = min(cur.shape[1], cx_ + w // 2), min(cur.shape[0], cy_ + h // 2)
        ax0, ay0 = max(0, cx_ - r), max(0, cy_ - r)
        ax1, ay1 = min(cur.shape[1], cx_ + r), min(cur.shape[0], cy_ + r)
        ring = np.ones((ay1 - ay0, ax1 - ax0), bool)
        ring[by0 - ay0:by1 - ay0, bx0 - ax0:bx1 - ax0] = False
        if ring.sum() < 20 or (by1 - by0) * (bx1 - bx0) < 9:
            continue
        c_in = float(np.median(lc_cur[by0:by1, bx0:bx1]))
        c_out = float(np.median(lc_cur[ay0:ay1, ax0:ax1][ring]))
        ratio = c_in / max(c_out, 1e-6)
        hits.append((long_side, w, h, float(np.abs(vals).mean()), cx_, cy_, ratio))
    hits.sort(key=lambda t_: t_[6])            # most transparent-looking first
    clear = [h_ for h_ in hits if h_[6] < 1.0]
    all_hits += len(hits)
    print(f"  f{t:04d}: sigma {sigma:5.1f}, {len(hits)} moving object(s) in the size "
          f"band, {len(clear)} with contrast BELOW their surroundings (transparent-like)")
    for long_side, w, h, lvl, x, y, ratio in hits[:5]:
        kind = "clear-like" if ratio < 1.0 else "glint-like"
        print(f"      {w:3d}x{h:3d}px @({x:4d},{y:4d})  contrast ratio {ratio:5.2f}  "
              f"{kind}   = {max(w, h) * Z / f_px:4.1f} mm at {Z:.0f} mm")

    if len(targets) <= 6 or VIDEO:
        base = S.to_u8(cur, lo=0, hi=float(np.percentile(cur, 99.5)))
        bgu = S.to_u8(bg, lo=0, hi=float(np.percentile(cur, 99.5)))
        # signed residual: grey is no change, bright is arrival, dark departure
        rs = np.clip(res / (K * sigma * 2.5), -1, 1)
        resu = ((rs + 1) * 127.5).astype(np.uint8)
        marked = np.dstack([base] * 3)
        for long_side, w, h, lvl, x, y, ratio in hits:
            col = (120, 190, 255) if ratio < 1.0 else (90, 90, 90)
            cv2.rectangle(marked, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2),
                          col, 1)
            if ratio < 1.0:
                cv2.putText(marked, f"{w}x{h} {ratio:.2f}", (x - w // 2, y - h // 2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        tiles = []
        for nm, img in (("frame", base), ("median background = what stood still", bgu),
                        ("residual" + (", lattice notched" if NOTCH and np.isfinite(period)
                                       else "") + " (grey = unchanged)", resu),
                        (f"{len(clear)} clear-like of {len(hits)} moving", marked)):
            t_ = img if img.ndim == 3 else np.dstack([img] * 3)
            for col, th in (((0, 0, 0), 4), ((235, 235, 235), 1)):
                cv2.putText(t_, nm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th,
                            cv2.LINE_AA)
            tiles.append(t_)
        row = np.hstack(tiles)
        if VIDEO:
            d = OUT / f"{BURST}_{S.cam_name(CAM)}_motion"
            d.mkdir(exist_ok=True)
            S.write_png(d / f"frame{t:04d}.png", row)
        else:
            rows.append(row)

if rows:
    strip = np.vstack(rows)
    foot = np.full((72, strip.shape[1], 3), 18, np.uint8)
    cv2.putText(foot, f"{BURST} {S.cam_name(CAM)}  median of {WIN} frames removed; "
                f"boxes are {px_lo * 0.7:.0f}-{px_hi * 1.6:.0f} px, i.e. "
                f"{min(MM):g}-{max(MM):g} mm around {Z:.0f} mm", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(foot, "the fence stands still and cancels; what is left moved - "
                "which is what a transparent capsule does and its refraction does not",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    name = f"{BURST}_{S.cam_name(CAM)}_motion_f{targets[0]:04d}.png"
    S.write_png(OUT / name, np.vstack([strip, foot]))
    print(f"\nwrote {OUT / name}")
if VIDEO:
    d = OUT / f"{BURST}_{S.cam_name(CAM)}_motion"
    v = S.encode_video(d, OUT / f"{BURST}_{S.cam_name(CAM)}_motion.mp4", 10)
    print(f"\nwrote {v}" if v else "\nno video written")
print(f"total candidates: {all_hits} over {len(targets)} frame(s)")
