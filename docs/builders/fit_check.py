"""Rough overflow check: full-width chars ~1 em, half-width ~0.55 em, line 1.25 em,
default insets 0.1 in left/right, 0.05 in top/bottom."""
import sys, unicodedata, math
from pptx import Presentation
p = Presentation(sys.argv[1])
def em(ch):
    return 1.0 if unicodedata.east_asian_width(ch) in "WF" else 0.55
for i, s in enumerate(p.slides, 1):
    for sh in s.shapes:
        if not sh.has_text_frame or not sh.text_frame.text.strip(): continue
        w = sh.width / 914400 - 0.2; h = sh.height / 914400 - 0.1
        need = 0; widest = 0
        for para in sh.text_frame.paragraphs:
            txt = "".join(r.text for r in para.runs)
            if not txt: continue
            size = max((r.font.size.pt for r in para.runs if r.font.size), default=18) / 72
            width = sum(em(c) for c in txt) * size
            widest = max(widest, width)
            need += max(1, math.ceil(width / w)) * size * 1.25
        bottom = (sh.top + sh.height) / 914400
        flag = []
        if need > h * 1.15: flag.append(f"height need {need:.2f} > box {h:.2f}")
        if (sh.top / 914400) + need > 10.9: flag.append(f"runs to y={sh.top/914400+need:.2f}")
        if flag:
            print(f"slide {i}: '{sh.text_frame.text[:28]}' " + "; ".join(flag))
print("checked")
