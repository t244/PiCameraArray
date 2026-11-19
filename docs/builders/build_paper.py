"""First draft of the system paper (English, .docx), numbers read from the result files."""
import json, sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH

M, F, OUT = sys.argv[1], sys.argv[2], sys.argv[3]          # results dir, figures dir, output
A, B = "20260728_073439", "20260729_053037"
rep = json.load(open(f"{M}/report_AB_f6.json"))["runs"]
f90 = {r: json.load(open(f"{M}/frames90_{r}.json")) for r in (A, B)}
ev = json.load(open(f"{M}/model_eval.json"))
fin = json.load(open("/workspace/code/output/final.json"))
pc = lambda x: f"{x * 100:.2f}"

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)


def para(text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    for part in split_todo(text):
        r = p.add_run(part[0]); r.bold = bold; r.italic = italic
        if size: r.font.size = Pt(size)
        if part[1]:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if align: p.alignment = align
    return p


def split_todo(text):
    """【...】 and [REF ...] spans are highlighted as open items."""
    out, buf, i = [], "", 0
    while i < len(text):
        for o, c in (("【", "】"), ("[REF", "]")):
            if text.startswith(o, i):
                j = text.find(c, i) + len(c)
                if buf: out.append((buf, False)); buf = ""
                out.append((text[i:j], True)); i = j
                break
        else:
            buf += text[i]; i += 1
    if buf: out.append((buf, False))
    return out


FIGNO = {}


def fig(name, width, caption):
    """Figures are numbered in the order they appear; any "Fig. N." written in the caption is replaced."""
    import re as _re
    FIGNO[name] = len(FIGNO) + 1
    caption = f"Fig. {FIGNO[name]}. " + _re.sub(r"^Fig\.\s*[0-9]+(\([a-z]\))?\.\s*", "", caption)
    doc.add_picture(f"{F}/{name}", width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = para(caption, size=9.5); p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def table(rows, caption, widths=None):
    para(caption, size=9.5, bold=True)
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.cell(i, j); cell.text = ""
            run = cell.paragraphs[0].add_run(str(v)); run.font.size = Pt(9.5); run.bold = (i == 0)
    doc.add_paragraph()


# ------------------------------------------------------------------ front ---
para("DRAFT for internal discussion, 2026-11-12. Not for distribution. Target journal: Measurement (Elsevier), first choice; Results in Engineering (Elsevier) as fallback. "
     "Figures marked 【JP】 are placeholders from the September report and will be redrawn in English. "
     "Moisture-estimation numbers are provisional: the material-absorption term of the model (eq. 3) is being corrected.",
     italic=True, size=9.5)
para("A synchronized multi-view near-infrared camera system for non-contact moisture monitoring "
     "of soft capsules inside a rotating tumble dryer", bold=True, size=16, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Tsuyoshi Takatani¹, 【Fuji Capsule co-author(s)】², Keiichi Zempo¹",
     size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
para("¹ University of Tsukuba   ² Fuji Capsule Co., Ltd. 【affiliation to confirm】", size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_heading("Abstract", level=1)
sA, sB = rep[A]["configs"]["warped, 4-view mean"], rep[B]["configs"]["warped, 4-view mean"]
para("Soft capsules are dried for many hours in rotating tumble dryers, and the shell moisture content is still checked by "
     "manual sampling and loss-on-drying (LOD) measurement. We present a measurement system that observes the capsule bed "
     "from outside the dryer's metal mesh and estimates moisture content without contact. The hardware is a 4×4 array of "
     "hardware-synchronized global-shutter cameras behind four nested near-infrared long-pass filters, with on-board storage "
     "for unattended runs of a full drying cycle. The software removes the dryer mesh by synthetic-aperture compositing of "
     "the calibrated views and temporal statistics, extracts band-wise radiometric levels, and estimates moisture with a "
     "per-product forward model of wet-surface appearance inverted under a monotone-drying constraint. In two full drying "
     f"runs of two products (21.9 h and 18.0 h, {fin['dataset'][A]['packed']} and {fin['dataset'][B]['packed']} bursts), "
     f"the mesh stripe energy was reduced by about 93 %, the 16 cameras stayed within ±1 frame, and the measurement "
     f"noise of the spectral levels was {pc(sA['sigma@120min'])} % and {pc(sB['sigma@120min'])} % over 2-h windows. "
     f"For capsule type A the leave-one-out moisture error was {ev[A]['loo']['model']:.2f} percentage points "
     f"against {ev[A]['loo']['clock']:.2f} for elapsed time alone 【provisional】; for capsule type B the "
     "near-infrared bands carried no moisture information, which motivates a visible band. We describe the system, its failure modes, "
     "and the requirements it sets for a production deployment.")
para("Keywords: soft capsule; drying; moisture content; camera array; synthetic aperture; near-infrared; process monitoring",
     italic=True, size=10)

# ------------------------------------------------------------- 1 intro ---
doc.add_heading("1. Introduction", level=1)
para("Soft gelatin capsules are filled, encapsulated and then dried in rotating tumble dryers before tray drying; the shell "
     "moisture content at the exit of the last tumbler decides whether a lot can proceed. In current practice operators take "
     "samples at intervals and measure moisture by LOD, which is slow, labour-intensive and sparse in time. A non-contact, "
     "continuous measurement would let the process be judged on every lot and at every moment. 【REF: soft capsule drying and "
     "moisture specification】")
para("Optical measurement inside a tumble dryer is difficult for four reasons: (i) the capsules are behind a rotating "
     "stainless-steel mesh that occludes a large part of every image; (ii) the capsule bed tumbles, so the visible surface "
     "changes completely within a second; (iii) the capsules are coated with lubricating oil and are glossy; and (iv) "
     "products differ in shell colour and transparency, so their optical response to drying differs.")
para("This paper contributes a complete measurement system addressing these conditions: (1) a 16-camera, hardware-"
     "synchronized near-infrared camera array with nested long-pass filters, designed to be mounted outside the dryer mesh "
     "and to record a full drying cycle unattended; (2) a processing pipeline that removes the mesh by synthetic-aperture "
     "compositing of calibrated views and extracts band-wise radiometric levels; and (3) a characterisation of the "
     "measurement noise and a physics-informed, per-product moisture model, evaluated against LOD references on two "
     "products. We also report where the system fails and what that implies for the next measurement campaign.")

# ------------------------------------------------------------- 2 related ---
doc.add_heading("2. Related work", level=1)
para("Moisture measurement in pharmaceutical drying. Near-infrared spectroscopy is established for granules and tablets "
     "【REF】. Acoustic emission has been used to estimate moisture during fluidized-bed drying of granules "
     "(Aoki et al., 2014; Aoki et al., 2022), and a companion acoustic system is being developed for the same dryers 【REF/plan】.")
para("Wet-surface appearance. Shimano et al. (2017) and Okawa et al. (2022) model the darkening and spectral sharpening of "
     "wet surfaces through multiple scattering and absorption, and recover wetness from a multispectral image. Our moisture "
     "model builds on this formulation.")
para("Camera arrays and synthetic aperture. Dense camera arrays synthesize a large aperture that defocuses foreground "
     "occluders 【REF: synthetic-aperture imaging with camera arrays】; de-fencing removes regular occluders using multiple "
     "views or motion 【REF: de-fencing】. Here the occluder is a rigid mesh close to the cameras and the scene moves, which "
     "allows both cross-view and temporal cues.")

# ------------------------------------------------------------- 3 system ---
doc.add_heading("3. Measurement system", level=1)
doc.add_heading("3.1 Hardware", level=2)
fig("s02_Image_0.jpg", 3.0, "Fig. 1(a). The 4×4 camera array in its enclosure. 【Photo to be updated so that all 16 cameras are visible】")
fig("s04_Image_1.jpg", 3.0, "Fig. 1(b). The array mounted at the side of the tumble dryer, outside the mesh.")
table([["Item", "Specification"],
       ["Cameras", "16 × Sony IMX296 monochrome global shutter, 1456 × 1088 px, 3.45 µm pixels, 10-bit raw"],
       ["Computers", "16 × Raspberry Pi 4B, one per camera; local SSD storage"],
       ["Synchronization", "External hardware trigger from a microcontroller; exposure = trigger pulse width (10 ms), 30 frames/s"],
       ["Geometry", "4 × 4 grid, ~40 mm pitch (~160 × 160 mm); 6 mm lenses (calibrated focal length ≈ 1796 px)"],
       ["Filters", "CLAREX NIR-70N / 80N / 85N / 90N long-pass (cut-on 670 / 760 / 810 / 860 nm), 2 × 2 tiling, 4 cameras each"],
       ["Illumination", "Halogen lamp, 40 W"],
       ["Working distance", "94–120 mm to the mesh, depending on grid row"],
       ["Acquisition", "Burst of 90 frames (2.97 s) every 2 min, for the full drying cycle"]],
      "Table 1. Hardware of the measurement system.")
fig("en_spectral.png", 5.8, "Fig. 2. Spectral response of the four long-pass filters and of the monochrome sensor. The filter "
    "curves are nominal (50 % at the stated cut-on wavelength) 【to be replaced with the manufacturer's transmission data】; the "
    "sensor response is read off the manufacturer's chart. Every channel integrates from its cut-on to the end of the sensor "
    "response, so the four channels are nested rather than separate bands.")
doc.add_heading("3.2 Acquisition and data handling", level=2)
para("Each camera runs a resident agent that switches between a preview mode for focusing and a triggered capture mode. "
     "A burst of 90 frames is stored as one file per camera on its local SSD. After a run, the files are collected over the "
     "network or by attaching the SSDs to the analysis server, verified by file count and size, and grouped per trigger into "
     "bursts of 16 views. Clock offsets between the computers do not matter for pairing because bursts are keyed by their "
     "trigger index; the residual frame offset between cameras was measured from the scene motion. 【describe failure "
     "handling: missed files, truncated files, wrong-session guard】")
doc.add_heading("3.3 Calibration", level=2)
para("Intrinsics and lens distortion of each camera and the relative pose of every camera pair were calibrated with a ChArUco "
     "board (5 × 7 squares). Views are warped onto a reference camera through the plane-induced homography at the depth of the "
     "capsule bed, with both lens models applied.")

# ------------------------------------------------------------- 4 methods ---
doc.add_heading("4. Processing", level=1)
doc.add_heading("4.1 Synthetic-aperture mesh removal", level=2)
fig("mosaic16_capsuleB.png", 6.0, "Fig. 3(a). The 16 raw views at one instant (capsule B, about 4 h into drying, frame 30 "
    "of a burst), laid out as the 4 × 4 array, each labelled with its camera and filter. Each view sees the mesh at a "
    "different position. Display scale is set per filter band and shared by the four views of that band.")
fig("en_principle.png", 6.3, "Fig. 3(b). Principle of mesh removal. (i) The camera array looks into the rotating tumble dryer "
    "through its steel mesh wall (illustration, near end plate cut away); rays from different cameras to the same capsule "
    "cross the mesh at different points. (ii) After the views are "
    "registered on the capsule plane, the capsules coincide while the mesh falls at a different place in each view. (iii) The "
    "mesh transmission of each view is estimated across views (the bright envelope) and over time (the bed moves, the mesh "
    "does not) and divided out.")
para("Within a band, the four views are warped to the capsule plane. The mesh transmission of each view is estimated from a "
     "levelled maximum across views and from a temporal upper envelope over 21 frames, and divided out; residual misregistration "
     "is corrected by block matching only when it is systematic. 【expand with equations】")
fig("defence_capsuleB_fig4.png", 5.8, "Fig. 4. Capsule B, about 4 h into drying (800 nm band, frame 30 of a burst), close-up "
    "of the same 560 × 400 px region in the reference view, chosen where the bed moves least: one raw view (left) and the "
    "result of mesh removal from the four views of the band (right), on one shared intensity scale and gamma (0.75). The "
    "bright blobs on the left repeat at the mesh pitch (53–55 px) and are mesh openings out of focus; the individual "
    "capsules are the smaller grains visible on the right.")
doc.add_heading("4.2 Band-wise radiometric levels", level=2)
para("For every burst, frames spread over the burst are read, black level is subtracted, and each view is warped onto a common "
     "reference view. The level of a band is the 95th percentile of the mean of its four warped views in a fixed region of "
     "interest, averaged in the log domain over the selected frames.")
doc.add_heading("4.3 Measurement noise", level=2)
para("We need a noise figure that is not confounded by the moisture change itself. Working with the logarithm of the four "
     "band levels, we remove two effects that carry no moisture information: a fixed offset per band (filter, sensor, "
     "illumination spectrum) and, at each instant, a factor common to all four bands (overall brightness, how much of the "
     "bed is in view). Of the remaining three degrees of freedom, water moves the levels along one known direction (Section "
     "4.4). The other two directions can be produced neither by water nor by a brightness change, so any variation in them "
     "is measurement noise. We report its root-mean-square per direction, in percent of intensity, as the measurement "
     "noise; lower is better. It is computed over the bursts of each run, either per burst or after averaging the bursts "
     "within 2-h windows.")
doc.add_heading("4.4 Moisture model", level=2)
para("We adapt the wet-surface model of Shimano et al. (2017):")
doc.add_picture(f"{F}/en_eq_shimano.png", width=Inches(4.8)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
fig("en_paths.png", 5.6, "Fig. 5. Wetting makes scattering more forward and paths longer, so each additional scattering event "
    "multiplies in both material (a) and water (b) absorption: the surface becomes darker and more saturated.")
para("Taking the number of scattering events as Poisson-distributed closes the sum, and we add a dye term that changes with "
     "drying, a soft knee from moisture content to surface wetness, and the nested filter integration:")
doc.add_picture(f"{F}/en_eq_model.png", width=Inches(4.8)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Each product has ten parameters. Oil absorbing above 860 nm projects onto the same direction as water in these "
     "channels and is therefore absorbed into the water path term; the output is the manufacturer's moisture content, "
     "not a photometric water amount. Moisture is recovered by dynamic programming on a moisture grid with the constraint "
     "that moisture does not increase during drying. 【eq. 3 is being revised so that u_j stays positive】")
fig("en_model_contrast.png", 5.6, "Fig. 6. What each term of the model does, with illustrative parameters, shown as the "
    "change of the ratio between the longest- and shortest-wavelength channels during drying. (a) Water absorption alone "
    "changes the ratio little and linearly. (b) Letting wetting lengthen the light paths amplifies the change and makes it "
    "stop once the surface is dry (below m*). (c) A shell dye whose absorption changes with drying alters the shape of the "
    "curve; since the dye differs between capsule types, so does the curve.")

# ------------------------------------------------------------- 5 exp ---
doc.add_heading("5. Experiments", level=1)
table([["", "Capsule A", "Capsule B"],
       ["Drying time", "21.9 h", "18.0 h"],
       ["Bursts (all 16 views)", f"{fin['dataset'][A]['packed']} ({fin['dataset'][A]['all16']})",
        f"{fin['dataset'][B]['packed']} ({fin['dataset'][B]['all16']})"],
       ["LOD references", "11 (every 2 h), 24.9 → 9.2 %", "9 (every 2 h), 21.5 → 8.8 %"],
       ["Oil cloth inserted", f"{fin['dataset'][A]['cloth']:.1f} h", f"{fin['dataset'][B]['cloth']:.1f} h"],
       ["Analysed window", f"0–{fin['dataset'][A]['cut']:.0f} h", f"0–{fin['dataset'][B]['cut']:.0f} h"]],
      "Table 2. Drying runs. Each run kept one lot in one tumbler for the whole drying; LOD measured by the manufacturer.")
fig("en_moisture.png", 5.6, "Fig. 7. LOD reference moisture of the two runs and the time the oil-absorbing cloth was inserted.")
para("The oil-absorbing cloth inserted near the end of drying covers the capsules; frames after it are excluded in this study. "
     "【this excludes the specification region near 8–8.5 %; to be addressed】")

# ------------------------------------------------------------- 6 results ---
doc.add_heading("6. Results", level=1)
doc.add_heading("6.1 Synchronization and mesh removal", level=2)
para("Frame offsets between cameras were within 0 ± 1 frame (±33 ms). Mesh removal reduced the periodic mesh component "
     "from 2.47 to 0.17 (about 93 %) and raised capsule brightness 2.9× (capsule A) and 2.2× (capsule B). 【re-measure with the "
     "current pipeline】")
doc.add_heading("6.2 Measurement noise", level=2)
names = [("raw, 1 camera", "One camera per filter, as captured"),
         ("raw, 4-view mean", "Four cameras per filter, as captured; levels averaged"),
         ("warped, 4-view mean", "Four cameras per filter, registered; levels averaged (proposed)"),
         ("warped, pixel-averaged", "Four cameras per filter, registered; images averaged before taking the level")]
rows = [["How the cameras of each filter are combined", "Capsule A, per burst", "Capsule A, 2-h average",
         "Capsule B, per burst", "Capsule B, 2-h average"]]
for k, lab in names:
    a, b = rep[A]["configs"][k], rep[B]["configs"][k]
    rows.append([lab, pc(a["sigma"]), pc(a["sigma@120min"]), pc(b["sigma"]), pc(b["sigma@120min"])])
table(rows, "Table 3. Measurement noise (%, RMS; lower is better; Section 4.3) for four ways of combining the cameras of "
      "each filter. All bursts of each run, 15 frames per burst. 'Registered' means warped onto a common reference view.")
fig("en_frames.png", 5.6, "Fig. 8. Measurement noise per burst against the number of frames averaged per burst, frames spread over the "
    "burst (every 8th burst, all 90 frames read).")
para(f"Averaging frames within a burst was the dominant factor: the noise fell from {f90[A]['1']['spread']*100:.1f} % with one "
     f"frame to {f90[A]['15']['spread']*100:.2f} % with 15 and {f90[A]['90']['spread']*100:.2f} % with 90 (capsule A), saturating "
     "beyond about 30 frames. Adjacent frames are nearly redundant, so frames should be spread over the burst.")
doc.add_heading("6.3 Ablations", level=2)
abl = json.load(open(f"{M}/ablation_views.json"))
para("(a) Number of views. Using only k of the four cameras of each band (all combinations averaged), the measurement noise falls "
     f"roughly as 1/√k: from {abl[A]['warp_1']['sigma']*100:.2f} % with one view per band (4 cameras) to "
     f"{abl[A]['warp_4']['sigma']*100:.2f} % with four (16 cameras) for capsule A, and from {abl[B]['warp_1']['sigma']*100:.2f} % "
     f"to {abl[B]['warp_4']['sigma']*100:.2f} % for capsule B. Registering the views onto one reference matters as much as their "
     f"number: over 2-h windows the noise for capsule B is {abl[B]['raw_4']['sigma2h']*100:.2f} % without warping and "
     f"{abl[B]['warp_4']['sigma2h']*100:.2f} % with it.")
fig("en_ablation_views.png", 5.8, "Fig. 10. Measurement noise against the number of views per band, with (solid) and without "
    "(dotted) registration onto a common reference view. Shaded: range over the camera combinations.")
para("(b) Number of spectral bands. 【to do: repeat the noise and the estimation with 2 and 3 of the 4 channels】")
para("(c) Mesh coverage. 【to do: measure the fraction of each view occluded by the mesh from the estimated transmission maps, "
     "and relate it to the noise per view】")
para(f"(d) Frames per burst: see Section 6.2 and Fig. {FIGNO['en_frames.png']}.")
doc.add_heading("6.4 Moisture estimation 【provisional; calibration results to be inserted after the planned measurements】", level=2)
fig("en_estimate.png", 5.6, "Fig. 9. LOD references and leave-one-out estimates; each estimate is made with that reference and "
    "the data within 1 h of it withheld.")
table([["Estimator (RMSE, percentage points)", "Capsule A", "Capsule B"],
       ["Image model (this work)", f"{ev[A]['loo']['model']:.2f}", f"{ev[B]['loo']['model']:.2f}"],
       ["Same, time order of observations shuffled (3 runs)", "1.47 / 2.82 / 29.3", "14.2 / 2.22 / 2.44"],
       ["Elapsed time, linear", f"{ev[A]['loo']['clock']:.2f}", f"{ev[B]['loo']['clock']:.2f}"],
       ["Elapsed time, exponential drying curve", "1.33", "0.70"]],
      "Table 4. Leave-one-out moisture error within each run.")

# ------------------------------------------------------------- 7 discussion ---
doc.add_heading("7. Discussion", level=1)
para("Differences between capsule types. The two capsule types differ in the optical properties of their shells "
     "【extent of description to confirm with the manufacturer】. For capsule A the image model outperformed elapsed time; "
     "for capsule B the near-infrared bands carried no moisture information, and a visible band is planned.")
para("Nested long-pass filters. Water, oil and illumination colour project onto nearly the same direction in the four "
     "channels, which limits what the spectral index can separate. 【decide how much of this belongs in the paper】")
para("Limitations. One lot per product, so the model has not been validated across lots; the specification region near the "
     "end of drying falls after the oil cloth is inserted; some model parameters are not identifiable from the present data. "
     "A simultaneous acoustic and image measurement over full drying cycles of several lots per product is planned. 【REF/plan】")

doc.add_heading("8. Conclusion", level=1)
para("We built and operated a synchronized near-infrared camera-array system that measures a tumbling capsule bed through "
     "the dryer mesh for a complete drying cycle, removes the mesh by synthetic-aperture compositing, and estimates moisture "
     "with a per-product physical model. 【to be written after the revised results】")

doc.add_heading("References", level=1)
for ref in [
    "M. Shimano, H. Okawa, Y. Asano, R. Bise, K. Nishino, I. Sato, Wetness and color from a single multispectral image, "
    "in: Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2017, pp. 3967–3975.",
    "H. Okawa, M. Shimano, Y. Asano, R. Bise, K. Nishino, I. Sato, Estimation of wetness and color from a single multispectral "
    "image, IEEE Trans. Pattern Anal. Mach. Intell. 44 (12) (2022) 8740–8753. doi:10.1109/TPAMI.2019.2903496.",
    "Aoki et al., 2014, RSC Advances. doi:10.1039/C4RA06433J. 【full reference to complete】",
    "Aoki et al., 2022, International Journal of Pharmaceutics. doi:10.1016/j.ijpharm.2022.121689. 【full reference to complete】",
    "[REF: synthetic-aperture imaging with camera arrays]",
    "[REF: image de-fencing]",
    "[REF: NIR moisture measurement of gelatin / soft capsules]"]:
    para(ref, size=9.5)

doc.save(OUT)
print("saved", OUT)
