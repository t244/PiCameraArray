"""One object from the gallery, rendered at several brightnesses.

The gallery stretches every tile on the 90-99th percentile band of its own
frame, which keeps the whole size ladder comparable but leaves a mid-level
object dim. This re-renders a single object over a ladder of mappings, so its
shape can be judged rather than merely located.

    python out/measure/bright.py burst0300 1 85 17x18      # the tile's label
    python out/measure/bright.py burst0300 1 85 x1163y742  # or its position

Brightening is for looking, not for measuring: raising the gain spreads a
blob's skirt above the threshold and it reads bigger than it is. The size in
the label is the one measured on the gallery's own band, unchanged.
"""
import json
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, "analyze")
import sa_packed as S                                            # noqa: E402

BURST = sys.argv[1] if len(sys.argv) > 1 else "burst0300"
CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FRAME = int(sys.argv[3]) if len(sys.argv) > 3 else 85
WANT = sys.argv[4] if len(sys.argv) > 4 else "17x18"
LEVEL = float(sys.argv[5]) if len(sys.argv) > 5 else None   # the label's "level"
BAND = (90.0, 99.0)
PATCH, MAG = 96, 8

out = Path("out/measure")
_, cams = S.discover_burst(Path("collected_data/packed/20260728_073439") / BURST)
calib = S.load_calibration("calibration_results.npz")
f_px = calib["cams"][CAM]["K"][0, 0]
p, pre = cams[CAM]

im, _, _ = S.load_camera_frame(p, pre, FRAME)
mx, my = S.sample_map(calib, CAM, CAM, 550.0, im.shape)
u = np.nan_to_num(S.remap_image(im, mx, my))
lo, hi = np.percentile(u, BAND)

# ---- find the object the label refers to ------------------------------------
if WANT.startswith("x"):
    cx, cy = (int(v) for v in WANT[1:].split("y"))
    w = h = 0
    mean = float(u[cy, cx])
else:
    want_w, want_h = (int(v) for v in WANT.lower().split("x"))
    m = (u > lo).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    # a frame holds several blobs of the same bounding box, so the label's
    # level is what tells them apart
    found = []
    for k in range(1, n):
        bw, bh = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        if (bw, bh) != (want_w, want_h):
            continue
        vals = u[lab == k]
        if vals.max() >= float(u.max()):
            continue
        found.append((k, bw, bh, float(vals.mean()), cen[k]))
    if not found:
        sys.exit(f"no {WANT} object in frame {FRAME}; run gallery.py to see the labels")
    if LEVEL is not None:
        best = min(found, key=lambda t: abs(t[3] - LEVEL))
    else:
        best = found[0]
        if len(found) > 1:
            print(f"  {len(found)} objects are {WANT}: levels "
                  f"{', '.join(f'{t[3]:.0f}' for t in found)} - "
                  f"add the level to choose, e.g. ... {WANT} {found[0][3]:.0f}")
    _, w, h, mean, c = best
    cx, cy = int(c[0]), int(c[1])

print(f"{BURST} {S.cam_name(CAM)} f{FRAME}: {w}x{h}px level {mean:.0f} at ({cx}, {cy})")
print(f"  frame band {BAND[0]:.0f}-{BAND[1]:.0f}th pct = {lo:.0f}-{hi:.0f} counts, "
      f"ceiling {u.max():.0f}")

y0 = max(0, min(u.shape[0] - PATCH, cy - PATCH // 2))
x0 = max(0, min(u.shape[1] - PATCH, cx - PATCH // 2))
patch = u[y0:y0 + PATCH, x0:x0 + PATCH]

# ---- a ladder of mappings ---------------------------------------------------
# The first is what the gallery does. Then the top of the range comes down to
# the object's own level and below it, which is what "brighter" means for a
# linear map. The last is a gamma curve: it lifts the mid-levels without
# pushing the object's core into the ceiling, so the outline stays readable.
views = [
    ("gallery", lambda a: S.to_u8(a, lo=lo * 0.4, hi=hi * 1.15)),
    (f"linear 0-{mean * 1.6:.0f}", lambda a: S.to_u8(a, lo=0.0, hi=mean * 1.6)),
    (f"linear 0-{mean:.0f}", lambda a: S.to_u8(a, lo=0.0, hi=mean)),
    ("gamma 0.5", lambda a: (255 * np.clip(
        np.maximum(a, 0) / max(mean * 2.0, 1e-6), 0, 1) ** 0.5).astype(np.uint8)),
]

tiles = []
for name, fn in views:
    t = cv2.resize(fn(patch), None, fx=MAG, fy=MAG,
                   interpolation=cv2.INTER_NEAREST)
    t = np.dstack([t] * 3)
    if w and h:
        cv2.rectangle(t, ((cx - x0 - w // 2) * MAG, (cy - y0 - h // 2) * MAG),
                      ((cx - x0 + w // 2) * MAG, (cy - y0 + h // 2) * MAG),
                      (110, 255, 110), 1)
    for col, th in (((0, 0, 0), 4), ((110, 255, 110), 1)):
        cv2.putText(t, name, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th,
                    cv2.LINE_AA)
    tiles.append(t)

strip = np.hstack(tiles)
foot = np.full((104, strip.shape[1], 3), 18, np.uint8)
cv2.putText(foot, f"{BURST} {S.cam_name(CAM)} frame {FRAME}: {w}x{h}px, level "
            f"{mean:.0f} counts, at ({cx},{cy}), {MAG}x nearest-neighbour",
            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 1, cv2.LINE_AA)
cv2.putText(foot, f"20mm subtends {20 * f_px / 550:.0f}px at 550mm, "
            f"{20 * f_px / 800:.0f}px at 800mm - this object is {max(w, h)}px, "
            f"i.e. 20mm at {20 * f_px / max(max(w, h), 1):.0f}mm",
            (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 170, 170), 1, cv2.LINE_AA)
cv2.putText(foot, "brighter only to see the shape: a raised gain widens the "
            "blob's skirt, so do not measure off these",
            (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (140, 140, 140), 1, cv2.LINE_AA)

name = f"{BURST}_{S.cam_name(CAM)}_f{FRAME:04d}_{w}x{h}_bright.png"
S.write_png(out / name, np.vstack([strip, foot]))

# Where every pixel of this strip came from, so a box drawn on it can be
# turned back into sensor pixels without anyone counting panels by hand.
(out / name).with_suffix(".json").write_text(json.dumps({
    "burst": BURST, "camera": S.cam_name(CAM), "frame": FRAME,
    "source_origin": [int(x0), int(y0)], "patch": PATCH, "mag": MAG,
    "panel_size": [PATCH * MAG, PATCH * MAG],
    "panels": [n for n, _ in views],
    "object": {"w": int(w), "h": int(h), "level": float(mean),
               "centre": [int(cx), int(cy)]},
    "plane_mm": 550.0, "f_px": float(f_px),
    "band_counts": [float(lo), float(hi)],
}, indent=2))
print(f"  wrote out/measure/{name}")
