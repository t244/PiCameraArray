"""Numbers for the report page, from the reports and level files."""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
rep = json.load(open("/workspace/out/moisture/report_AB_f6.json"))
B = F.BANDS
mu = np.array([fin["mumodel"]["mu"][b] for b in B]); c = mu - mu.mean()
pl = np.array([0.0] + [fin["lamp"]["planck"][b] for b in B[1:]]); pl = pl - pl.mean()
unit = lambda v: v / np.linalg.norm(v)
out = {"runs": {}, "old": {k: fin["floors"][k] for k in fin["floors"]}}
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}
for run, r in rep["runs"].items():
    s = r["series"]; t = np.array(s["hours"]); w = np.array(s["water"]); m = np.array(s["moisture"]); z = np.array(s["null"])
    e = np.arange(0, t.max() + .5, .5); i = np.digitize(t, e)
    g = lambda x: [round(float(x[i == k].mean()), 5) for k in np.unique(i)]
    ds, ref = fin["dataset"][run], fin["reference"][run]
    rows = [x for x in F.load("/workspace/out/moisture/" + files[run]) if (int(x["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    tt, L = F.levels(rows, "warp", "p95")
    X = L - L.mean(1, keepdims=True); X = X - X.mean(0, keepdims=True)
    mm = F.moisture_at(tt, ref)
    d = np.array([np.polyfit(mm, x, 1)[0] for x in X])
    out["runs"][run] = {
        "product": ds["product"], "cut": ds["cut"], "bursts": r["bursts"],
        "bins": {"t": g(t), "water": g(w), "moisture": g(m), "null0": g(z[0]), "null1": g(z[1])},
        "configs": r["configs"], "stats": r["stats"], "sampling": r["sampling"], "frames": r["frames"],
        "lamp_removed": r["lamp_removed"],
        "dir": {"d": (d * 100).round(4).tolist(), "cos_water": float(unit(d) @ unit(-c)),
                "cos_lamp": float(unit(d) @ unit(pl)), "norm": float(np.linalg.norm(d) * 100),
                "water_slope": float(np.polyfit(np.array(g(m)), np.array(g(w)), 1)[0])},
        "reference": ref}
out["cos_water_lamp"] = float(unit(-c) @ unit(pl))
out["water_dir"] = (unit(-c)).round(4).tolist()
json.dump(out, open("/workspace/out/moisture/page_data.json", "w"))
print("ok", {k: (v["dir"]["water_slope"], v["dir"]["cos_water"]) for k, v in out["runs"].items()})
