import sys
from pptx import Presentation
from pptx.util import Emu
p = Presentation(sys.argv[1])
print("slide size", p.slide_width, p.slide_height, "in:", p.slide_width/914400, p.slide_height/914400)
print("masters:", [m.name if hasattr(m,'name') else '' for m in p.slide_masters])
for mi, m in enumerate(p.slide_masters):
    print(" master", mi, "layouts:", [l.name for l in m.slide_layouts])
for i, s in enumerate(p.slides, 1):
    print(f"\n=== slide {i} layout={s.slide_layout.name}")
    for sh in s.shapes:
        t = sh.shape_type
        pos = f"({sh.left/914400 if sh.left is not None else 0:.2f},{sh.top/914400 if sh.top is not None else 0:.2f}) {sh.width/914400 if sh.width else 0:.2f}x{sh.height/914400 if sh.height else 0:.2f}in"
        txt = ""
        if sh.has_text_frame:
            runs = []
            for para in sh.text_frame.paragraphs:
                r = "".join(x.text for x in para.runs)
                if r.strip():
                    f = para.runs[0].font if para.runs else None
                    sz = f.size.pt if f is not None and f.size else None
                    runs.append(f"[lvl{para.level} {sz}pt] {r}")
            txt = " | ".join(runs)
        ph = f" ph={sh.placeholder_format.idx}/{sh.placeholder_format.type}" if sh.is_placeholder else ""
        print(f"  - {t} {sh.name}{ph} {pos} {txt[:300]}")
        if sh.has_table:
            tb = sh.table
            for r in tb.rows:
                print("      row:", [c.text for c in r.cells])
