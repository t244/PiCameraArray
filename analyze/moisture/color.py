"""NIR colour of each product: band ratios on the same cameras, same exposure."""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
B = F.BANDS
mu = np.array([fin["mumodel"]["mu"][b] for b in B]); c = mu - mu.mean(); u = c / np.linalg.norm(c)
pl = np.array([0.0] + [fin["lamp"]["planck"][b] for b in B[1:]]); pl -= pl.mean(); pl /= np.linalg.norm(pl)
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}
prof = {}
for run, f in files.items():
    ds, ref = fin["dataset"][run], fin["reference"][run]
    rows = [r for r in F.load("/workspace/out/moisture/" + f) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    # same grid row on both runs: row 0 cameras for 750/800, row 1 for 850/900 (nearest the drum)
    near = {b: [c_ for c_ in F.CAMS[b] if F.ROW[c_] in (0, 1)] for b in B}
    t, L = F.levels(rows, "raw", "p95", cams=near)
    m = F.moisture_at(t, ref)
    early, late = L[:, m >= m.max() - 0.5].mean(1), L[:, m <= m.min() + 1.5].mean(1)
    prof[run] = (early, late, m.max(), m.min())
    r = lambda v: "  ".join(f"{b}/750 {np.exp(v[i]-v[0]):.3f}" for i, b in enumerate(B) if i)
    print(f"{run} ({ds['product']}): wet ({m.max():.1f}%) {r(early)}")
    print(f"{' '*27} dry ({m.min():.1f}%) {r(late)}")
    d = late - early; d = d - d.mean()
    print(f"{' '*27} wet->dry change, common removed: " + " ".join(f"{x*100:+.2f}%" for x in d),
          f"| cos water {(-d/np.linalg.norm(d)) @ u:+.2f}  cos lamp {(-d/np.linalg.norm(d)) @ pl:+.2f}")
(eA, lA, *_), (eB, lB, *_) = prof.values()
diff = eB - eA; diff -= diff.mean()
print("colour difference aka - shizuku (wet, log, common removed):", " ".join(f"{x*100:+.1f}%" for x in diff),
      f"| cos with water dir {diff/np.linalg.norm(diff) @ u:+.2f}")
