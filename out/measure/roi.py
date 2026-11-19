"""What a box drawn on one of these pictures actually is.

Give it the strip that bright.py wrote and the rectangle you dragged on it,
in that picture's own pixels:

    python out/measure/roi.py out/measure/burst0300_e01_f0085_17x18_bright.png 2649,277,140,252

The strip's json says which panel that lands in, at what magnification, and
where the patch came from, so the box goes back to sensor pixels exactly.
A box in sensor pixels can be given directly instead:

    python out/measure/roi.py burst0300 1 85 883,274,18,32

It reports the size, what it would be at a range of distances, and - the part
worth reading - how far the blob extends at each threshold. A bright object
sitting on a bright structure has no single size: the number you get is the
one your threshold chose, which is why the eye and the detector disagree.
"""
import json
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, "analyze")
import sa_packed as S                                            # noqa: E402

PACKED = Path("collected_data/packed/20260728_073439")
MAG_OUT = 12

# What this session was shooting, and roughly how far away. burst0300 is the
# teardrop run: 10-15 mm, highly transparent. The 20 mm capsules are the other
# run (the 541-file one), so do not carry that number over.
TARGET_MM = [10.0, 12.5, 15.0]
TARGET_Z = 700.0

opts = {}
for a in sys.argv[1:]:
    if "=" in a and not a.endswith(".png"):
        k, v = a.split("=", 1)
        opts[k] = v
argv = [a for a in sys.argv if "=" not in a or a.endswith(".png")]
sys.argv = argv
if "z" in opts:
    TARGET_Z = float(opts["z"])
if "mm" in opts:
    TARGET_MM = [float(x) for x in opts["mm"].split(",")]


def from_strip(png: Path, rect):
    meta = json.loads(png.with_suffix(".json").read_text())
    rx, ry, rw, rh = rect
    pw = meta["panel_size"][0]
    panel = rx // pw
    ox, oy = meta["source_origin"]
    mag = meta["mag"]
    sx = ox + (rx - panel * pw) / mag
    sy = oy + ry / mag
    print(f"strip   : {png.name}, panel {panel} "
          f"({meta['panels'][panel] if panel < len(meta['panels']) else '?'}), {mag}x")
    return (meta["burst"], int(meta["camera"][1:]), meta["frame"],
            (sx, sy, rw / mag, rh / mag), meta)


if sys.argv[1].endswith(".png"):
    png = Path(sys.argv[1])
    rect = [int(v) for v in sys.argv[2].replace("x", ",").split(",")]
    burst, cam, frame, (sx, sy, sw, sh), meta = from_strip(png, rect)
else:
    burst, cam, frame = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sx, sy, sw, sh = (float(v) for v in sys.argv[4].split(","))
    meta = {}

_, cams = S.discover_burst(PACKED / burst)
calib = S.load_calibration("calibration_results.npz")
f_px = calib["cams"][cam]["K"][0, 0]
p, pre = cams[cam]
im, _, _ = S.load_camera_frame(p, pre, frame)
mx, my = S.sample_map(calib, cam, cam, meta.get("plane_mm", 550.0), im.shape)
u = np.nan_to_num(S.remap_image(im, mx, my))

x0, y0 = int(round(sx)), int(round(sy))
x1, y1 = int(round(sx + sw)), int(round(sy + sh))
roi = u[y0:y1, x0:x1]
ccx, ccy = (x0 + x1) // 2, (y0 + y1) // 2

print(f"source  : {burst} {S.cam_name(cam)} frame {frame}")
print(f"box     : ({sx:.1f}, {sy:.1f})  {sw:.1f} x {sh:.1f} px, centre ({ccx}, {ccy})")
print(f"levels  : max {roi.max():.0f}  mean {roi.mean():.0f}  median {np.median(roi):.0f}"
      f"   frame median {np.median(u):.0f}, 99th pct {np.percentile(u, 99):.0f}, "
      f"ceiling {u.max():.0f}")

print("\nhow far it reaches, per threshold (the component under the box centre):")
rows = []
for thr in (np.percentile(u, [75, 90, 95, 99]).tolist()
            + [roi.max() * 0.5, roi.max() * 0.7]):
    m = (u > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    k = lab[ccy, ccx]
    if k == 0:
        rows.append((thr, None))
        print(f"  > {thr:5.0f} counts : the centre is below this")
        continue
    w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
    rows.append((thr, (w, h, st[k, cv2.CC_STAT_AREA])))
    note = "  <- merged into the surrounding structure" if max(w, h) > 4 * max(sw, sh) else ""
    print(f"  > {thr:5.0f} counts : {w:4d} x {h:4d} px, area {st[k, cv2.CC_STAT_AREA]:6d}{note}")

long_px, short_px = max(sw, sh), min(sw, sh)
print(f"\nsize, against a {TARGET_MM[0]:.0f}-{TARGET_MM[-1]:.0f} mm target at "
      f"{TARGET_Z:.0f} mm (f = {f_px:.0f} px):")
for L in TARGET_MM:
    print(f"  {L:5.1f} mm would span {L * f_px / TARGET_Z:5.1f} px at {TARGET_Z:.0f} mm"
          f"   - this box is {long_px:.1f} px, i.e. that size at {L * f_px / long_px:5.0f} mm")
print(f"  at {TARGET_Z:.0f} mm the box measures "
      f"{long_px * TARGET_Z / f_px:5.2f} x {short_px * TARGET_Z / f_px:4.2f} mm, "
      f"aspect {long_px / short_px:.2f}")

# ---- transparent-object cues ------------------------------------------------
# A clear capsule has no bright edge to threshold on. What it does have: it
# refracts what is behind it, so the background texture inside its outline is
# displaced and softened, and its curved surface throws one small specular
# core. Comparing the box against a ring around it says whether that is what
# is here, or whether this is simply a bright speck.
def local_contrast(a, k=9):
    a = a.astype(np.float32)
    mu = cv2.blur(a, (k, k))
    mu2 = cv2.blur(a * a, (k, k))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    return sd / np.maximum(mu, 1.0)

ring = int(round(max(sw, sh)))
ax0, ay0 = max(0, x0 - ring), max(0, y0 - ring)
ax1, ay1 = min(u.shape[1], x1 + ring), min(u.shape[0], y1 + ring)
around = np.ones((ay1 - ay0, ax1 - ax0), bool)
around[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] = False
lc = local_contrast(u)
box_lc, ring_lc = lc[y0:y1, x0:x1], lc[ay0:ay1, ax0:ax1][around]
box_lv, ring_lv = u[y0:y1, x0:x1], u[ay0:ay1, ax0:ax1][around]
print("\ntransparent or just bright? box vs the ring around it:")
print(f"  level          {np.median(box_lv):7.0f}  vs {np.median(ring_lv):7.0f} counts"
      f"   (peak in box {box_lv.max():.0f})")
print(f"  local contrast {np.median(box_lc):7.3f}  vs {np.median(ring_lc):7.3f}"
      f"   ratio {np.median(box_lc) / max(np.median(ring_lc), 1e-6):.2f}")
print(f"  darkest rim px {box_lv.min():7.0f}  vs {np.percentile(ring_lv, 5):7.0f}"
      f"   (a refracting edge often reads darker than its surroundings)")
print("  a clear capsule reads as softened background texture (contrast ratio "
      "below ~1) with one small bright core;")
print("  an opaque speck reads as higher level AND higher contrast than the ring.")

# ---- picture: the box on its surroundings, at three brightnesses -----------
pad = int(max(sw, sh))
px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
px1, py1 = min(u.shape[1], x1 + pad), min(u.shape[0], y1 + pad)
patch = u[py0:py1, px0:px1]
lvl = float(roi.mean())
lc_patch = lc[py0:py1, px0:px1]
views = [("as captured (90-99 pct)", S.to_u8(patch, lo=np.percentile(u, 90) * 0.4,
                                            hi=np.percentile(u, 99) * 1.15)),
         (f"linear 0-{lvl * 1.6:.0f}", S.to_u8(patch, lo=0.0, hi=lvl * 1.6)),
         ("gamma 0.5", (255 * np.clip(np.maximum(patch, 0) / max(lvl * 2, 1e-6),
                                      0, 1) ** 0.5).astype(np.uint8)),
         ("local contrast 9px", S.to_u8(lc_patch, lo_pct=2, hi_pct=98))]
tiles = []
for nm, v in views:
    t = np.dstack([cv2.resize(v, None, fx=MAG_OUT, fy=MAG_OUT,
                              interpolation=cv2.INTER_NEAREST)] * 3)
    cv2.rectangle(t, ((x0 - px0) * MAG_OUT, (y0 - py0) * MAG_OUT),
                  ((x1 - px0) * MAG_OUT, (y1 - py0) * MAG_OUT), (110, 255, 110), 2)
    # the target sizes at the assumed distance, on the same scale
    yy = t.shape[0] - 24
    xx = 24
    for L, col in zip(TARGET_MM, ((120, 190, 255), (110, 255, 110), (255, 200, 110))):
        d = int(L * f_px / TARGET_Z * MAG_OUT)
        cv2.circle(t, (xx + d // 2, yy - d // 2), d // 2, col, 2, cv2.LINE_AA)
        cv2.putText(t, f"{L:g}mm", (xx, yy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
        xx += d + 24
    for col, th in (((0, 0, 0), 4), ((110, 255, 110), 1)):
        cv2.putText(t, nm, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th, cv2.LINE_AA)
    tiles.append(t)
strip = np.hstack(tiles)
foot = np.full((88, strip.shape[1], 3), 18, np.uint8)
cv2.putText(foot, f"{burst} {S.cam_name(cam)} f{frame}  box ({sx:.0f},{sy:.0f}) "
            f"{sw:.0f}x{sh:.0f}px at {MAG_OUT}x  =  "
            f"{long_px * TARGET_Z / f_px:.1f} x {short_px * TARGET_Z / f_px:.1f} mm "
            f"if it sits at {TARGET_Z:.0f} mm", (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 1, cv2.LINE_AA)
cv2.putText(foot, "circles are the target sizes at that distance. A clear capsule has "
            "no edge to threshold: look for softened background and one specular core",
            (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 150, 150), 1, cv2.LINE_AA)
name = f"{burst}_{S.cam_name(cam)}_f{frame:04d}_roi{int(sx)}x{int(sy)}.png"
S.write_png(Path("out/measure") / name, np.vstack([strip, foot]))
print(f"\nwrote out/measure/{name}")
