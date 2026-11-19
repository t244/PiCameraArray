"""16 raw views of one instant, laid out as the 4x4 array, one display scale per filter band."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import sa_packed as S

burst, frame, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
scale, G = 0.36, 0.75
FILTER = {750: "NIR-70N", 800: "NIR-80N", 850: "NIR-85N", 900: "NIR-90N"}
_, cams = S.discover_burst(burst)
imgs = {i: S.load_camera_frame(*cams[i], frame)[0] for i in range(16)}
rng = {}
for b in S.BANDS:                       # one scale per band, shared by its four views
    v = np.concatenate([imgs[i].ravel()[::13] for i in range(16) if S.cam_band(i) == b])
    rng[b] = (np.percentile(v, 1), np.percentile(v, 99.5))
tiles = {}
for i, im in imgs.items():
    lo, hi = rng[S.cam_band(i)]
    u8 = (np.clip((im - lo) / (hi - lo), 0, 1) ** G * 255).astype(np.uint8)
    u8 = cv2.resize(u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = f"{S.cam_name(i)}  {FILTER[S.cam_band(i)]}"
    cv2.rectangle(u8, (0, 0), (len(lab) * 13 + 14, 34), 0, -1)
    cv2.putText(u8, lab, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)
    tiles[i] = u8
h, w = tiles[0].shape
gap = 8
sheet = np.full((4 * h + 3 * gap, 4 * w + 3 * gap), 255, np.uint8)
for i, t in tiles.items():
    r, c = S.cam_rc(i)
    sheet[r * (h + gap): r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = t
cv2.imwrite(out, sheet)
print("bands:", {b: (round(float(l), 1), round(float(u), 1)) for b, (l, u) in rng.items()},
      "| layout row/col of e00..e03:", [S.cam_rc(i) for i in range(4)], "| sheet", sheet.shape)
