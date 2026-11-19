"""sigma = spread of the null space left after removing common + water (+ lamp)?"""
import json, numpy as np
O = "/workspace/out_old/"
fin = json.load(open(O + "final.json"))
B = ("750", "800", "850", "900")
mu = np.array([fin["mumodel"]["mu"][b] for b in B])
mul = np.array([fin["mumodel"]["mu_label"][b] for b in B])
pl = np.array([0.0] + [fin["lamp"]["planck"][b] for b in B[1:]])

def basis_null(dirs):
    A = np.array(dirs).T                                   # 4 x k
    q, _ = np.linalg.qr(np.c_[A, np.eye(4)])
    return q[:, len(dirs):4]                               # orthonormal complement

def cands(levels, m, extra):
    L = np.log(np.array([levels[b] for b in B], float))    # 4 x T
    L = L - L.mean(1, keepdims=True)                       # per band over the run
    L = L - L.mean(0, keepdims=True)                       # per time across bands
    N = basis_null([np.ones(4), m] + extra)
    z = N.T @ L
    v = z.var(1)
    return {"rms": np.sqrt(v.mean()), "total": np.sqrt(v.sum()), "first": np.sqrt(v[0]),
            "last": np.sqrt(v[-1]), "rms_ddof1": np.sqrt(z.var(1, ddof=1).mean())}

for run in ("20260728_073439", "20260729_053037"):
    sv = json.load(open(O + f"saverdict_{run}_v2.json"))
    for stage in ("raw", "warped", "defenced", "masked"):
        lv = sv["stages"][stage]["levels"]
        for lab, m, tgt in (("mu", mu, sv["stages"][stage]["sigma"]),
                            ("label", mul, fin["floors_label"][run][stage]["sigma"])):
            for xl, extra in (("", []), ("+lamp", [pl])):
                c = cands(lv, m, extra)
                k, v = min(c.items(), key=lambda kv: abs(kv[1] - tgt))
                print(run[:8], stage, lab + xl, "target", round(tgt, 4), "best", k, round(v, 4))
