import sys, re
from pptx import Presentation
p = Presentation(sys.argv[1])
def fill(sh):
    try:
        f = sh.fill
        if f.type == 1: return "solid " + str(f.fore_color.rgb if f.fore_color.type == 1 else f.fore_color.theme_color)
        return str(f.type)
    except Exception as e: return "?"
def line(sh):
    try:
        l = sh.line
        return f"{l.color.rgb if l.fill.type==1 and l.color.type==1 else l.fill.type} {l.width.pt if l.width else ''}"
    except Exception: return "?"
for i in [1, 3, 12, 20]:
    s = p.slides[i-1]
    print(f"=== slide {i}  bg: {s.background.fill.type if s.background else None}")
    for sh in list(s.shapes)[:16]:
        info = f"{sh.name:14s} fill={fill(sh)} line={line(sh)}"
        if sh.has_text_frame and sh.text_frame.text.strip():
            r = next((r for pa in sh.text_frame.paragraphs for r in pa.runs if r.text.strip()), None)
            if r is not None:
                f = r.font
                col = None
                try: col = f.color.rgb if f.color and f.color.type == 1 else (f.color.theme_color if f.color and f.color.type else None)
                except Exception: col = "?"
                ea = r._r.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
                lat = r._r.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}latin')
                info += f" | text='{sh.text_frame.text[:30]}' size={f.size.pt if f.size else None} bold={f.bold} color={col} latin={lat.get('typeface') if lat is not None else None} ea={ea.get('typeface') if ea is not None else None} align={sh.text_frame.paragraphs[0].alignment}"
        print("  ", info)
# theme fonts and colors
import zipfile
z = zipfile.ZipFile(sys.argv[1])
th = [n for n in z.namelist() if n.startswith('ppt/theme/theme')]
x = z.read(th[0]).decode('utf8')
print("theme fonts:", re.findall(r'<a:(?:latin|ea) typeface="([^"]*)"', x)[:6])
print("theme colors:", re.findall(r'<a:(dk1|lt1|dk2|lt2|accent\d)>.*?(?:val|lastClr)="([0-9A-F]{6})"', x))
bg = z.read('ppt/slides/slide1.xml').decode('utf8')
print("slide1 bg xml:", re.findall(r'<p:bg>.*?</p:bg>', bg)[:1])
