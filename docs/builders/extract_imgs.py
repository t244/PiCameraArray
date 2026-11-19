import sys
from pptx import Presentation
p = Presentation(sys.argv[1]); out = sys.argv[2]
for i, s in enumerate(p.slides, 1):
    for sh in s.shapes:
        if sh.shape_type == 13:
            im = sh.image
            name = f"s{i:02d}_{sh.name.replace(' ', '_')}.{im.ext}"
            open(f"{out}/{name}", "wb").write(im.blob)
            print(name, im.size, f"{sh.width/914400:.2f}x{sh.height/914400:.2f}in", "crop", sh.crop_left, sh.crop_right, sh.crop_top, sh.crop_bottom)
