"""Which detrending of the water-direction index reproduces saverdict's sigma?"""
import json, numpy as np
O = "/workspace/out_old/"
fin = json.load(open(O + "final.json"))
mu = np.array([fin["mumodel"]["mu"][b] for b in ("750", "800", "850", "900")])
mul = np.array([fin["mumodel"]["mu_label"][b] for b in ("750", "800", "850", "900")])

def index(levels, m):
    L = np.log(np.array([levels[b] for b in ("750", "800", "850", "900")], float))
    c = m - m.mean(); c /= np.linalg.norm(c)
    return c @ L

def cands(s, t):
    out = {"std": s.std(), "std1": s.std(ddof=1), "diff/rt2": np.diff(s).std() / np.sqrt(2),
           "mad*1.4826": 1.4826 * np.median(np.abs(s - np.median(s)))}
    for deg in (1, 2, 3, 4, 5):
        r = s - np.polyval(np.polyfit(t, s, deg), t); out[f"poly{deg}"] = r.std()
    for w in (3, 5, 7, 9, 11, 15, 21, 31):
        k = np.ones(w) / w; pad = np.pad(s, w // 2, mode="edge")
        out[f"box{w}"] = (s - np.convolve(pad, k, "valid")).std()
        med = np.array([np.median(pad[i:i + w]) for i in range(len(s))])
        out[f"med{w}"] = (s - med).std()
    return out

for run in ("20260728_073439", "20260729_053037"):
    sv = json.load(open(O + f"saverdict_{run}_v2.json")); t = np.array(sv["hours"])
    for stage in ("raw", "warped", "defenced", "masked"):
        lv = sv["stages"][stage]["levels"]
        for lab, m, tgt in (("mu", mu, sv["stages"][stage]["sigma"]),
                            ("label", mul, fin["floors_label"][run][stage]["sigma"])):
            c = cands(index(lv, m), t)
            best = sorted(c.items(), key=lambda kv: abs(kv[1] - tgt))[:3]
            print(run[:8], stage, lab, "target", round(tgt, 4), [(k, round(v, 4)) for k, v in best])
