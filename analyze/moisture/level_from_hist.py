"""Which histogram statistic reproduces saverdict's per-burst level?"""
import json, numpy as np
O = "/workspace/out_old/"
sl = json.load(open(O + "salev_20260728_073439_v2.json"))
sv = json.load(open(O + "saverdict_20260728_073439_v2.json"))
binw = sl["binw"]; centres = (np.arange(sl["nbin"]) + 0.5) * binw
print("salev bursts", len(sl["rows"]), sl["rows"][0]["burst"], sl["rows"][1]["burst"], sl["rows"][-1]["burst"])
print("verdict hours[:4]", sv["hours"][:4], "n", len(sv["hours"]))
for stage in ("raw", "warped", "defenced", "masked"):
    for band in ("750", "900"):
        h = np.array(sl["rows"][0]["bands"][band][stage], float)
        target = sv["stages"][stage]["levels"][band][0]
        c = np.cumsum(h) / h.sum()
        cand = {"mean": (h * centres).sum() / h.sum(), "median": np.interp(.5, c, centres)}
        for q in (90, 95, 98, 99, 99.5, 99.9):
            cand[f"p{q}"] = np.interp(q / 100, c, centres)
        for top in (1, 2, 5, 10, 20):
            thr = np.interp(1 - top / 100, c, centres); m = centres >= thr
            cand[f"top{top}mean"] = (h[m] * centres[m]).sum() / h[m].sum()
        best = sorted(cand.items(), key=lambda kv: abs(kv[1] - target))[:3]
        print(stage, band, "target", round(target, 3), [(k, round(v, 3)) for k, v in best])
