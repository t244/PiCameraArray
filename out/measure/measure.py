"""A measuring sheet: 1:1 crops with a pixel grid and size references.

Nothing is resized, so a pixel on screen is a pixel on the sensor. The grid is
every 50 px; the reference circles are what a 20 mm object subtends at a range
of depths, so a capsule can be matched against them by eye.
"""
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "analyze"); import sa_packed as S

BURST = sys.argv[1] if len(sys.argv) > 1 else "burst0500"
CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FRAMES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                           else "10,25,40,55,70,85".split(","))]
X, Y, W, H = (int(v) for v in (sys.argv[4].split(",") if len(sys.argv) > 4 else "380,180,900,620".split(",")))

out = Path("out/measure"); out.mkdir(parents=True, exist_ok=True)
_, cams = S.discover_burst(Path("collected_data/packed/20260728_073439") / BURST)
calib = S.load_calibration("calibration_results.npz")
c = calib["cams"][CAM]
f_px = c["K"][0, 0]
p, pre = cams[CAM]

def undistorted(idx):
    im, _, _ = S.load_camera_frame(p, pre, idx)
    mx, my = S.sample_map(calib, CAM, CAM, 550.0, im.shape)
    return np.nan_to_num(S.remap_image(im, mx, my))

# one range for the whole set, so the frames are comparable
ref = undistorted(FRAMES[len(FRAMES)//2])
# Stretched to the usable middle of the range, not to its extremes: the top
# 0.4% of every frame is clipped and carries no shape, and the bottom is the
# read noise of a scene whose median is fourteen counts. Mapping 0.4x to 1.15x
# of the 90-99th percentile band puts what can actually be measured on screen.
b_lo, b_hi = np.percentile(ref, [90, 99])
lo, hi = b_lo * 0.4, b_hi * 1.15
period, _ = S.fence_period_px(ref)
print(f"{BURST} {S.cam_name(CAM)} ({S.cam_band(CAM)}nm)  f = {f_px:.0f} px")
print(f"mesh period measured here: {period:.1f} px")
print(f"stretched to {lo:.0f}-{hi:.0f} counts (the 90-99th percentile band, "
      f"ceiling {ref.max():.0f})")
print(f"a 20 mm object subtends: " + ", ".join(
    f"{20*f_px/z:.0f}px at {z}mm" for z in (300, 400, 550, 700, 900)))

DEPTHS = [(300, (255, 120, 120)), (400, (255, 200, 110)), (550, (110, 255, 110)),
          (700, (120, 190, 255)), (900, (200, 140, 255))]

def annotate(u8):
    img = np.dstack([u8] * 3)
    g = img.copy()
    for x in range(0, W, 50):
        cv2.line(g, (x, 0), (x, H), (70, 70, 70), 1)
    for y in range(0, H, 50):
        cv2.line(g, (0, y), (W, y), (70, 70, 70), 1)
    for x in range(0, W, 250):
        cv2.line(g, (x, 0), (x, H), (120, 120, 120), 1)
        cv2.putText(g, f"{x}", (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
    for y in range(0, H, 250):
        cv2.line(g, (0, y), (W, y), (120, 120, 120), 1)
        cv2.putText(g, f"{y}", (4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
    img = cv2.addWeighted(img, 0.75, g, 0.25, 0)
    return img

def rule_strip():
    """The reference circles, on their own strip so they hide nothing."""
    s = np.full((190, W, 3), 18, np.uint8)
    cv2.putText(s, "a 20 mm object at:", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (230, 230, 230), 1, cv2.LINE_AA)
    x = 30
    for z, col in DEPTHS:
        d = 20 * f_px / z
        cx, cy = int(x + d / 2), 110
        cv2.circle(s, (cx, cy), int(d / 2), col, 1, cv2.LINE_AA)
        cv2.putText(s, f"{z}mm", (cx - 26, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
        cv2.putText(s, f"{d:.0f}px", (cx - 24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
        x += d + 34
    if np.isfinite(period):
        cv2.line(s, (int(x) + 20, 110), (int(x + period) + 20, 110),
                 (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(s, f"mesh {period:.0f}px", (int(x) + 6, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(s, f"= {period*100/f_px:.1f}mm at 100mm", (int(x) + 6, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return s

strip = rule_strip()
sheets = []
for idx in FRAMES:
    full = undistorted(idx)
    u = full[Y:Y+H, X:X+W]
    u8 = S.tone_map(S.to_u8(u, lo=lo, hi=hi), 1.4, 0.0)
    img = annotate(u8)
    # the same crop with the fence's own lattice notched out, so a capsule is
    # not being read through a grid of the same size as itself
    nf = S.notch_fence(full, period)[Y:Y+H, X:X+W]
    nb = np.percentile(nf, [90, 99])
    n8 = S.tone_map(S.to_u8(nf, lo=nb[0] * 0.4, hi=nb[1] * 1.15), 1.4, 0.0)
    nimg = annotate(n8)
    cv2.putText(nimg, "same crop, fence lattice notched out", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(nimg, "same crop, fence lattice notched out", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.vstack([img, nimg])
    cv2.putText(img, f"{BURST} {S.cam_name(CAM)} frame {idx}  1:1, no resizing",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"{BURST} {S.cam_name(CAM)} frame {idx}  1:1, no resizing",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.vstack([img, strip])
    S.write_png(out / f"{BURST}_{S.cam_name(CAM)}_x{X}y{Y}_f{idx:04d}.png", sheet)
    sheets.append(cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
rows = [np.hstack(sheets[i:i+2]) for i in range(0, len(sheets) - 1, 2)]
S.write_png(out / f"{BURST}_{S.cam_name(CAM)}_x{X}y{Y}_contact.png", np.vstack(rows))
print("wrote", len(FRAMES), "sheets +", f"{BURST}_{S.cam_name(CAM)}_contact.png", "to", out)
