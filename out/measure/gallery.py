"""The objects in the middle of the brightness range, magnified, next to what
20 mm would look like.

The extremes are not measurable. About 0.4% of every frame sits exactly at the
sensor ceiling, and a clipped blob's edges are where the clipping stopped, not
where the object did - which is why the brightest things look big and shapeless.
At the other end, the frame's median is fourteen counts, so anything near it is
read noise wearing a shape. What can be measured is in between, and that is all
this selects.
"""
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "analyze"); import sa_packed as S

BURST = sys.argv[1] if len(sys.argv) > 1 else "burst0500"
CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 1
BAND = tuple(float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                                else "90,99".split(",")))
FRAMES = [10, 25, 40, 55, 70, 85]
MAG, PATCH = 3, 110
out = Path("out/measure"); out.mkdir(parents=True, exist_ok=True)
_, cams = S.discover_burst(Path("collected_data/packed/20260728_073439") / BURST)
calib = S.load_calibration("calibration_results.npz")
f_px = calib["cams"][CAM]["K"][0, 0]
p, pre = cams[CAM]

def view(idx):
    im, _, _ = S.load_camera_frame(p, pre, idx)
    mx, my = S.sample_map(calib, CAM, CAM, 550.0, im.shape)
    return np.nan_to_num(S.remap_image(im, mx, my))

def objects(u, band):
    """Blobs whose own level sits inside the band and that are not clipped."""
    lo, hi = np.percentile(u, band)
    ceiling = float(u.max())
    m = (u > lo).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    keep, clipped, toobright = [], 0, 0
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        if a < 12 or a > 20000:
            continue
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        # compact and roughly round: a snake of glints run together at this
        # threshold is not one object, and measuring its length answers
        # nothing about how big a capsule is
        if a / float(w * h) < 0.50 or max(w, h) / max(min(w, h), 1) > 2.0:
            continue
        vals = u[lab == k]
        if vals.max() >= ceiling:
            clipped += 1; continue
        if vals.mean() > hi:
            toobright += 1; continue
        keep.append((k, a, float(vals.mean()), st[k], cen[k]))
    return keep, (lo, hi), clipped, toobright

rows, sizes, tot = [], [], [0, 0, 0]
for idx in FRAMES:
    u = view(idx)
    keep, (lo, hi), clipped, toobright = objects(u, BAND)
    keep_all = keep
    tot[0] += len(keep)
    tot[1] += clipped; tot[2] += toobright
    for _, a, _, st, _ in keep_all:
        sizes.append((2 * (a / np.pi) ** 0.5,
                      max(st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT])))
    # a ladder across the size distribution, not just its tail: one object
    # near each of these percentiles, so the whole range is on show
    keep.sort(key=lambda t: t[1])
    if keep:
        picks = [keep[min(len(keep) - 1, int(q / 100 * len(keep)))]
                 for q in (50, 75, 90, 97, 99.5)]
    else:
        picks = []
    keep_all, keep = keep, picks
    # stretched to the band, so a mid-brightness object is not a grey smudge
    u8 = S.to_u8(u, lo=lo * 0.4, hi=hi * 1.15)
    tiles = []
    for k, a, mean, st, cen in keep:
        cx, cy = int(cen[0]), int(cen[1])
        y0 = max(0, min(u8.shape[0] - PATCH, cy - PATCH // 2))
        x0 = max(0, min(u8.shape[1] - PATCH, cx - PATCH // 2))
        t = cv2.resize(u8[y0:y0 + PATCH, x0:x0 + PATCH], None, fx=MAG, fy=MAG,
                       interpolation=cv2.INTER_NEAREST)
        t = np.dstack([t] * 3)
        w, h = st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT]
        cv2.rectangle(t, ((cx - x0 - w // 2) * MAG, (cy - y0 - h // 2) * MAG),
                      ((cx - x0 + w // 2) * MAG, (cy - y0 + h // 2) * MAG),
                      (110, 255, 110), 1)
        for col, th in (((0, 0, 0), 3), ((110, 255, 110), 1)):
            cv2.putText(t, f"f{idx} {w}x{h}px  level {mean:.0f}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) < 5:
        tiles.append(np.zeros((PATCH * MAG, PATCH * MAG, 3), np.uint8))
    rows.append(np.hstack(tiles))

ref = np.full((PATCH * MAG, PATCH * MAG * 5, 3), 18, np.uint8)
cv2.putText(ref, f"same {MAG}x magnification: a 20mm object at", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
cv2.putText(ref, f"band {BAND[0]:.0f}-{BAND[1]:.0f}th pct ({lo:.0f}-{hi:.0f} counts), "
            f"clipped excluded. Columns are the 50th, 75th, 90th, 97th and 99.5th "
            f"percentile of size", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (170, 170, 170), 1, cv2.LINE_AA)
x = 40
for z, col in ((300, (255, 120, 120)), (400, (255, 200, 110)), (550, (110, 255, 110)),
               (700, (120, 190, 255)), (900, (200, 140, 255))):
    d = 20 * f_px / z * MAG
    cv2.circle(ref, (int(x + d / 2), PATCH * MAG // 2 + 20), int(d / 2), col, 2, cv2.LINE_AA)
    cv2.putText(ref, f"{z}mm {20*f_px/z:.0f}px", (int(x), PATCH * MAG - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    x += d + 40
S.write_png(out / f"{BURST}_{S.cam_name(CAM)}_gallery.png", np.vstack(rows + [ref]))

s = np.array(sizes)
print(f"{BURST} {S.cam_name(CAM)}: band = {BAND[0]:.0f}-{BAND[1]:.0f}th percentile "
      f"({lo:.0f}-{hi:.0f} counts, ceiling {view(FRAMES[0]).max():.0f})")
print(f"  {tot[0]} objects in the band, {tot[1]} rejected as clipped, "
      f"{tot[2]} as brighter than the band")
for q in (25, 50, 75, 90, 99):
    print(f"  {q:2d}th pct: equivalent diameter {np.percentile(s[:,0], q):5.1f} px, "
          f"longest side {np.percentile(s[:,1], q):5.1f} px")
for name, px in (("median", np.median(s[:,1])), ("90th pct", np.percentile(s[:,1], 90)),
                 ("99th pct", np.percentile(s[:,1], 99))):
    print(f"  a {name} object of {px:.0f} px is 20 mm at {20*f_px/px:6.0f} mm, "
          f"or {px*550/f_px:5.1f} mm if it is at 550 mm")
band_lo, band_hi = 20*f_px/800, 20*f_px/400          # 20 mm seen from 800 to 400 mm
n_in = int(((s[:,1] >= band_lo) & (s[:,1] <= band_hi)).sum())
print(f"  compact objects {band_lo:.0f}-{band_hi:.0f} px, i.e. 20 mm somewhere between "
      f"400 and 800 mm: {n_in} of {len(s)} ({n_in/len(s)*100:.1f}%), "
      f"{n_in/len(FRAMES):.0f} per frame")
