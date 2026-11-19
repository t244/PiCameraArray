"""Ablation: views per band (1..4), raw vs warped, all combinations averaged."""
import json, sys, itertools
import numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
mu = np.array([fin["mumodel"]["mu"][b] for b in F.BANDS])
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}
out = {}
for run, f in files.items():
    ds = fin["dataset"][run]
    rows = [r for r in F.load("/workspace/out/moisture/" + f) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    out[run] = {}
    for src in ("raw", "warp"):
        for k in (1, 2, 3, 4):
            res = []
            for combo in itertools.combinations(range(4), k):
                cams = {b: [F.CAMS[b][i] for i in combo] for b in F.BANDS}
                t, L = F.levels(rows, src, "p95", cams=cams)
                fl = F.floor(t, L, mu)[0]
                res.append((fl["sigma"], fl["sigma@120min"]))
            res = np.array(res)
            out[run][f"{src}_{k}"] = {"sigma": float(res[:, 0].mean()), "sigma_min": float(res[:, 0].min()),
                                       "sigma_max": float(res[:, 0].max()), "sigma2h": float(res[:, 1].mean()),
                                       "n_combos": len(res)}
            o = out[run][f"{src}_{k}"]
            print(f"{run[:8]} {src:4s} {k} view(s)/band ({4*k:2d} cameras): per-burst {o['sigma']*100:.2f}% "
                  f"[{o['sigma_min']*100:.2f}-{o['sigma_max']*100:.2f}]  2-h {o['sigma2h']*100:.2f}%", flush=True)
json.dump(out, open("/workspace/out/moisture/ablation_views.json", "w"), indent=1)
