import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); E = 914400
for i, s in enumerate(p.slides, 1):
    pics = [sh for sh in s.shapes if sh.shape_type == 13]
    texts = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
    for pc in pics:
        px0, py0, px1, py1 = pc.left/E, pc.top/E, (pc.left+pc.width)/E, (pc.top+pc.height)/E
        print(f"slide {i}: picture {px0:.2f},{py0:.2f} → {px1:.2f},{py1:.2f}")
        if px1 > 20.0 or py1 > 11.25: print("   OFF SLIDE")
        for t in texts:
            tx0, ty0, tx1, ty1 = t.left/E, t.top/E, (t.left+t.width)/E, (t.top+t.height)/E
            if tx0 < px1 and tx1 > px0 and ty0 < py1 and ty1 > py0:
                print(f"   overlaps text '{t.text_frame.text[:24]}' {tx0:.2f},{ty0:.2f} → {tx1:.2f},{ty1:.2f}")
