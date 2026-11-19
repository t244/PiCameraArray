"""Floor against frames per burst, on the every-8th-burst / all-90-frames subset."""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
mu = np.array([fin["mumodel"]["mu"][b] for b in F.BANDS])
run, path = sys.argv[1], sys.argv[2]
ds = fin["dataset"][run]
rows = [r for r in F.load(path) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
print(run, len(rows), "bursts")
res = {}
for n in (1, 2, 3, 5, 9, 15, 30, 45, 90):
    spread = set(np.linspace(0, 89, n).round().astype(int).tolist())
    block = set(range(n))
    a = F.floor(*F.levels(rows, "warp", "p95", frames=spread), mu)[0]
    b = F.floor(*F.levels(rows, "warp", "p95", frames=block), mu)[0]
    res[n] = {"spread": a["sigma"], "block": b["sigma"], "spread_slow": a["slow"]}
    print(f"  {n:3d} frames  spread over burst {a['sigma']*100:5.2f}%   first-n block {b['sigma']*100:5.2f}%")
json.dump(res, open(f"/workspace/out/moisture/frames90_{run}.json", "w"))
