"""Build the 2026-11-12 meeting deck on the 2026-09-04 deck's master and style."""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

SRC, OUT, IMG = sys.argv[1], sys.argv[2], sys.argv[3]
FONT = "Meiryo UI"
NO_NOTES = True          # the deck goes out without speaker notes
TITLE, DEEP, CARD, BODY, MUTED, PAGE, GOLD, WHITE = (
    "92278F", "632E62", "EAE5EB", "222222", "666666", "6E1D6B", "FCC704", "FFFFFF")

prs = Presentation(SRC)
# drop the old slides, keep the master, layouts and theme
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)
LAYOUT = prs.slide_layouts[0]


def rgb(h):
    return RGBColor.from_string(h)


def _font(run, size, bold, color):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    run.font.name = FONT
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT)


def text(slide, x, y, w, h, paras, size=21, bold=False, color=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None):
    """paras: str, or list of paragraphs; a paragraph is str or list of (text, overrides)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            _font(r, o.get("size", size), o.get("bold", bold), o.get("color", color))
    return tb


def rect(slide, x, y, w, h, color):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(color)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def set_notes(slide, body):
    """Speaker notes. The 09/04 deck's notes master has no body placeholder, so
    add one by hand; PowerPoint shows a body placeholder in the notes pane."""
    if NO_NOTES:
        return
    ns = slide.notes_slide
    tf = ns.notes_text_frame
    if tf is None:
        from pptx.oxml import parse_xml
        sp = parse_xml(
            '<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
            '<p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>'
            '<p:spPr><a:xfrm><a:off x="685800" y="4400550"/><a:ext cx="5486400" cy="3600450"/></a:xfrm></p:spPr>'
            '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="ja-JP"/></a:p></p:txBody></p:sp>')
        ns.shapes._spTree.append(sp)
        tf = ns.notes_text_frame
    tf.text = body


def new_slide(title, lead=None, page=None, notes=None):
    s = prs.slides.add_slide(LAYOUT)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(WHITE)
    text(s, 0.21, 0.31, 20.85, 0.66, title, size=49.5, bold=True, color=TITLE)
    if lead:
        text(s, 0.83, 1.48, 18.33, 0.68, lead, size=31.5, bold=True, color=DEEP)
    if page is not None:
        text(s, 19.16, 10.58, 0.6, 0.36, str(page), size=18, color=PAGE, align=PP_ALIGN.RIGHT)
    if notes:
        set_notes(s, notes)
    return s


def table(slide, y, cols, header, rows, size=21, row_h=0.62, first_bold=True):
    """cols: list of (x, w); header: list of str; rows: list of (cells, height or None)."""
    x0, x1 = cols[0][0] - 0.21, cols[-1][0] + cols[-1][1] + 0.1
    rect(slide, x0, y, x1 - x0, 0.69, DEEP)
    for (x, w), h in zip(cols, header):
        text(slide, x, y + 0.13, w, 0.44, h, size=size, bold=True, color=WHITE)
    y += 0.69
    for cells, h in rows:
        h = h or row_h
        for i, ((x, w), c) in enumerate(zip(cols, cells)):
            bold = first_bold and i == 0
            text(slide, x, y + 0.12, w, h - 0.14, c, size=size, bold=bold, color=DEEP if bold else BODY)
        y += h
        rect(slide, x0, y, x1 - x0, 0.02, CARD)
    return y


def picture(slide, name, x, y, w=None, h=None, caption=None):
    """Place an image by width or height (aspect kept); optional muted caption below."""
    pic = slide.shapes.add_picture(f"{IMG}/{name}", Inches(x), Inches(y),
                                   width=Inches(w) if w else None, height=Inches(h) if h else None)
    if caption:
        text(slide, x, y + pic.height / 914400 + 0.05, pic.width / 914400, 0.4, caption, size=18, color=MUTED)
    return pic


def card(slide, x, y, w, h, label, big, desc, big_size=48):
    rect(slide, x, y, w, h, CARD)
    text(slide, x + 0.42, y + 0.41, w - 0.6, 0.41, label, size=21, bold=True, color=TITLE)
    text(slide, x + 0.42, y + 1.0, w - 0.6, 0.96, big, size=big_size, bold=True, color=DEEP)
    text(slide, x + 0.42, y + 2.05, w - 0.7, h - 2.3, desc, size=27, color=BODY)


# ---------------------------------------------------------------- 1 cover ---
s = prs.slides.add_slide(LAYOUT)
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(WHITE)
text(s, -0.30, 3.37, 20.60, 2.24,
     [[("富士カプセル御中 打合せ資料", {"size": 66})], [("音響と画像の同時計測に向けて", {"size": 60})]],
     bold=True, color=TITLE, align=PP_ALIGN.CENTER)
rect(s, 5.26, 5.77, 9.48, 0.12, GOLD)
text(s, 1.75, 6.46, 16.50, 1.04, "髙谷 剛志", size=48, bold=True, color="000000", align=PP_ALIGN.CENTER)
text(s, 1.75, 7.54, 16.50, 0.69, "筑波大学システム情報系・助教", size=36, color="000000", align=PP_ALIGN.CENTER)
text(s, 1.75, 8.30, 16.50, 0.69, "2026/11/12", size=36, color="000000", align=PP_ALIGN.CENTER)
text(s, 19.16, 10.58, 0.6, 0.36, "1", size=18, color=PAGE, align=PP_ALIGN.RIGHT)
set_notes(s, 
    "本日は，画像による含水率推定の現状を短くご報告し，その上で，音響と合同で行う追加計測の内容と日程をご相談したいと考えています。"
    "最後に，計測の準備に必要な確認事項をお願いします。")

# --------------------------------------------------------------- 2 agenda ---
s = new_slide("本日のご相談", "画像の現状をご報告し，音響と画像の同時計測の内容と日程を決めたい", 2,
              "本日は3点です。画像の現状，同時計測の内容と日程，確認事項です。2点目の同時計測が一番ご相談したいところです。")
card(s, 0.83, 2.89, 5.86, 4.83, "1 画像の現状", "ばらつき 1/10", "全データで解析し直し，測定のばらつきが大きく減少")
card(s, 7.07, 2.89, 5.86, 4.83, "2 同時計測", "11〜12月", "一つのタンブラで，乾燥の初めから最後まで音と画像を同時に取る")
card(s, 13.31, 2.89, 5.86, 4.83, "3 ご確認事項", "3点", "潤滑油の種類，吸油布を入れる時刻の記録，11 月の計測日の候補")

# ---------------------------------------------------------------- 3 tests ---
s = new_slide("これまでの画像試験", "音響とは別のタンブラー1台で，1つのロットの乾燥の全過程を撮影", 3,
              "7月に行った画像の試験は2回です。音響が録音したタンブラーとは別の1台に，1つのロットを乾燥の最初から最後まで入れて撮りました。"
              "含水率は御社に2時間おきに測っていただいた値で，音響と同じ測り方です。")
cols = [(1.04, 3.3), (4.5, 2.6), (7.25, 2.6)]
table(s, 2.4, cols, ["項目", "しずく 7/28", "赤 7/29"], [
    (["撮影時間", "21.9 時間", "18.0 時間"], None),
    (["含水率（御社の実測）", "24.9 → 9.2 %", "21.5 → 8.8 %"], None),
    (["実測点（2 時間おき）", "11 点", "9 点"], None),
    (["吸油布の投入", "19.5 時間後", "14.4 時間後"], None),
    (["布投入前の含水率", "24.9 → 10.2 %", "21.5 → 10.4 %"], None),
    (["16 台が揃った撮影", "656 / 658 回", "537 / 541 回"], None),
])
text(s, 0.83, 7.25, 9.0, 1.0, "16 台のカメラ・4 種のフィルタで，2 分ごとに約 3 秒（90 コマ）撮影．含水率は音響と同じ方法（御社で計測）",
     size=21, color=MUTED)
picture(s, "s05_Image_0.png", 10.3, 2.4, w=9.0, caption="御社提供の含水率実測値（2 時間おき）")

# ---------------------------------------------------------------- 4 floor ---
s = new_slide("わかったこと① 測定のばらつきは 1/10 以下に",
              "データを間引かずにすべて使い，1 回の撮影の中のコマも平均した", 4,
              "9月にご報告した測定のばらつき，5.0%と4.3%は，データを間引いて計算していました。すべてのデータを使うと，2時間ごとの平均で0.3〜0.4%まで下がります。"
              "しずくの乾燥全体での変化の10分の1以下で，ノイズはもう大きな問題ではなくなりました。定義は9月と同じです。")
card(s, 0.83, 2.4, 7.4, 3.2, "しずく", "5.0 % → 0.40 %", "9月 → 今回（2 時間ごとの平均）", big_size=54)
card(s, 0.83, 5.85, 7.4, 3.2, "赤", "4.3 % → 0.33 %", "9月 → 今回（2 時間ごとの平均）", big_size=54)
picture(s, "fig_frames.png", 8.6, 2.4, w=10.6)
text(s, 8.75, 8.1, 10.4, 1.9, [
    "9月は，撮影を 1 回おきに，1 回の撮影から 6 コマだけ使っていた",
    "1 回の撮影から 30 コマ程度を平均すれば十分（それ以上はほぼ改善しない）"], size=22, spacing=8)

# ------------------------------------------------------------- 5 accuracy ---
s = new_slide("わかったこと② しずくは経過時間より正確に推定",
              "油や色素の変化も含めて含水率を出す推定式を，製品ごとに作成", 5,
              "目的は物理量としての水の量ではなく工程での含水率なので，潤滑油や色素の変化も工程の一部として推定式に含めました。"
              "しずくでは画像による推定が経過時間だけの予測を上回りました。時間の順番をばらばらにした対照では精度が出ないので，画像そのものに情報があります。"
              "赤では，今の近赤外の波長からは情報が取れていません。乾燥で赤みが増すので，可視光を加えることを考えています。"
              "どちらも1つのロットの中での評価で，ロットを入れ替えた評価にはロットを増やす必要があります。")
cols = [(1.04, 4.6), (5.85, 1.5), (7.55, 1.5)]
y = table(s, 2.4, cols, ["誤差（%ポイント）", "しずく", "赤"], [
    (["画像による推定", "1.12", "2.41"], None),
    (["経過時間だけ（直線）", "1.25", "1.44"], None),
    (["経過時間だけ（乾燥曲線）", "1.33", "0.70"], None),
])
rect(s, 0.83, 5.25, 8.5, 1.75, CARD)
text(s, 1.17, 5.4, 7.9, 0.45, "しずく（透明）", size=21, bold=True, color=TITLE)
text(s, 1.17, 5.9, 7.9, 1.0, "光がカプセルの集まりの奥まで届き，乾燥に伴う変化が見えている．経過時間による予測を上回った", size=21)
rect(s, 0.83, 7.2, 8.5, 1.75, CARD)
text(s, 1.17, 7.35, 7.9, 0.45, "赤（不透明）", size=21, bold=True, color=TITLE)
text(s, 1.17, 7.85, 7.9, 1.0, "今の近赤外の波長では手がかりがない．乾燥で増す赤みを捉える可視光が必要", size=21)
picture(s, "fig_estimate.png", 9.7, 2.3, w=9.5)
text(s, 0.83, 9.15, 18.33, 1.2,
     "誤差は，同じ試験の中で実測点を 1 点ずつ伏せて予測した結果（9月の「片方の試験で作った式を，もう片方に当てはめる」評価とは異なる）．"
     "製品ごとに 1 ロットのため，ロットを入れ替えた評価はまだできていない", size=18, color=MUTED)

# ----------------------------------------------------------------- 6 gaps ---
s = new_slide("課題：今のデータでは音響と比べられない",
              "タンブラー7の出口での判定に向けて，足りないものが3つある", 6,
              "音響側は，タンブラー7の出口で含水率が規格の8.0〜8.5%に入ったかを判定することを目指しています。"
              "画像の数字をこれと並べるには3つ足りません。同じロット・採取点がないこと，製品ごとに1ロットしかないこと，規格の境目が吸油布を入れた後の区間にあることです。"
              "吸油布は製造に必要と伺っているので，布が入った状態でも撮れるようにする必要があります。")
card(s, 0.83, 2.35, 5.86, 3.8, "1 別のタンブラー", "採取点が共通でない",
     "音響と画像を合わせた推定を確かめる材料がない", big_size=36)
card(s, 7.07, 2.35, 5.86, 3.8, "2 ロットの数", "製品ごとに 1 ロット",
     "別のロットに当てはまるかを確かめられない（音響は 9 ロット）", big_size=36)
card(s, 13.31, 2.35, 5.86, 3.8, "3 規格の境目", "吸油布の後の区間",
     "8.0〜8.5 % 付近は布がカプセルを隠すため，今は解析できていない", big_size=36)
picture(s, "s07_Picture_3.png", 0.83, 6.4, w=9.4, caption="吸油布を入れると画像が暗くなる（9月のご報告より）")
text(s, 10.6, 6.6, 8.6, 3.4, [
    [("吸油布は製造に必要 → 布が入った状態でも撮れる計測が要る", {"size": 24})],
    [("→ 音響と画像を合わせた推定を確かめるには，", {}), ("同時計測が必要", {"bold": True, "color": DEEP})]],
     size=27, spacing=18)

# ------------------------------------------------------------- 7 proposal ---
s = new_slide("ご提案：音響と画像の同時計測", "一つのタンブラで，乾燥の初めから最後までを音響と同時に計測したい", 7,
              "一つのタンブラにカメラとマイクを並べて付け，1つのロットを乾燥の初めから最後まで，音と画像で同時に計測したいと考えています。"
              "ロットは製品ごとに3つ以上あると，別のロットに当てはまるかを確かめられます。"
              "時刻を合わせるため，採取点の番号と時刻を1つの台帳に記録し，撮影機器と録音機の時計も計測の前後で合わせます。"
              "9月にご提案したランプ交換などもこの計測に合わせて行います。可視光のフィルタは，時期によっては間に合わない可能性があります。")
cols = [(1.04, 2.0), (3.2, 10.9)]
picture(s, "s04_Image_1.jpg", 14.45, 2.4, h=5.0, caption="7月の試験で取り付けたカメラ")
table(s, 2.4, cols, ["項目", "内容"], [
    (["1 設置", "カメラとマイクを同じタンブラに並べて設置"], 0.8),
    (["2 計測", "一つのタンブラで，乾燥の初めから最後までを音と画像で同時に計測"], 0.8),
    (["3 ロット", "製品ごとに 3 ロット以上"], 0.8),
    (["4 台帳", "採取点の番号・時刻・ロット・設備番号・吸油布の投入時刻を記録"], 0.8),
    (["5 時刻", "撮影機器と録音機の時計を，計測の前後に合わせる"], 0.8),
    (["6 装置", "光源交換・フィルタ並べ替え・感度校正を実施"], 0.8),
    (["7 波長", "（時期しだい）赤みの変化を捉える可視光のフィルタを追加"], 0.8),
])

# ------------------------------------------------------------- 8 schedule ---
s = new_slide("日程案：同時計測は 11〜12 月に", "共同研究は 2027 年 3 月末まで．2 月の比較から逆算する", 8,
              "共同研究は来年3月末までです。カメラの設置方法などの検討は11月中に済ませ，同時計測は11月から12月に行います。"
              "1月に採取点の番号と時刻の対応，音響と画像の出力の形をそろえ，2月に音響だけ，画像だけ，統合の3つを同じ条件で比べます。")
cols = [(1.04, 3.6), (4.9, 14.0)]
table(s, 2.4, cols, ["時期", "内容"], [
    (["11 月 12 日", "本日：同時計測の内容と日程を決める"], None),
    (["11 月", "カメラの設置方法の検討，フィルタの手配，台帳の様式づくり"], None),
    (["11〜12 月", "音響と画像の同時計測（製品ごとに 3 ロット以上）"], None),
    (["1 月", "採取点の番号・時刻の対応と，音響・画像の出力の形をそろえる"], None),
    (["2 月", "音響だけ・画像だけ・統合を，同じ条件で比べる"], None),
    (["3 月", "最終評価．精度を決めている要因と，実運用に向けた整理"], None),
])

# ------------------------------------------------------------ 9 questions ---
s = new_slide("ご確認・ご協力のお願い", "計測の準備のために 3 点", 9)
cols = [(1.04, 8.6), (10.0, 9.0)]
table(s, 2.4, cols, ["お伺いしたいこと", "理由"], [
    (["潤滑油の種類", "油の変化も含めて推定するため，どの油かを把握しておきたい"], 1.0),
    (["吸油布を入れる時刻の記録", "布が入った後の区間を，解析で正しく扱うため"], 1.0),
    (["11 月の計測日の候補", "音響と画像の同時計測の日程を決めるため"], 1.0),
], first_bold=False)

# ------------------------------------------------------------ 10 paper ---
s = new_slide("論文化の案", "回転タンブラ内のソフトカプセルの含水率を計測するシステムとして論文化する", 10)
cols = [(1.04, 3.2), (4.45, 14.6)]
table(s, 2.4, cols, ["項目", "案"], [
    (["主題", "合成開口と分光フィルタを用いた，回転タンブラ内のソフトカプセル含水率計測システム（ハードウェアとソフトウェア）"], 1.0),
    (["貢献", "① 金網の外から乾燥の全工程を無人で撮影する，同期 16 台の近赤外カメラアレイ"], 0.75),
    (["", "② 合成開口による金網除去と，分光による計測処理"], 0.75),
    (["", "③ 測定のばらつきの評価と，濡れた表面の見え方に基づく含水率モデル"], 0.75),
    (["投稿先", "Measurement（Elsevier）を第一候補，控えに Results in Engineering（Elsevier）"], 0.75),
    (["著者の順（案）", "髙谷 剛志，富士カプセル ご担当者（ご相談），善甫（筑波大学）"], 0.75),
    (["下書き", "図入りの英文の下書きを別添（推定精度の数値は暫定）"], 0.75),
])

# ============================================================ appendix ===
def arrow(slide, x, y, w, h):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = rgb(TITLE); a.line.fill.background(); a.shadow.inherit = False
    return a


# ---------------------------------------------------- 10 appendix cover ---
s = prs.slides.add_slide(LAYOUT)
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(WHITE)
text(s, -0.30, 3.6, 20.60, 1.2, "補足資料", size=66, bold=True, color=TITLE, align=PP_ALIGN.CENTER)
rect(s, 5.26, 5.0, 9.48, 0.12, GOLD)
text(s, 1.75, 5.5, 16.50, 0.8, "含水率推定モデルの詳細（Shimano らの式に基づく）", size=36, color="000000", align=PP_ALIGN.CENTER)
text(s, 19.16, 10.58, 0.6, 0.36, "11", size=18, color=PAGE, align=PP_ALIGN.RIGHT)

# ------------------------------------------------- 11 Shimano et al. ---
s = new_slide("出発点：濡れた表面の見え方",
              "Shimano ら（CVPR 2017）：濡れると暗くなるだけでなく，色が濃くなることを式で表した", 12)
picture(s, "fig_paths.png", 0.83, 2.4, w=10.0)
picture(s, "eq_shimano.png", 11.1, 2.45, w=8.2)
text(s, 11.2, 4.5, 8.0, 3.6, [
    [("n", {"bold": True, "color": DEEP}), ("：表面の中で光が散乱する回数", {})],
    [("a(λ)", {"bold": True, "color": DEEP}), ("：散乱 1 回あたりの材質の吸収（＝色）", {})],
    [("b(λ)", {"bold": True, "color": DEEP}), ("：散乱 1 回あたりの水の吸収", {})],
    [("γ", {"bold": True, "color": DEEP}), ("：濡れの度合い（0：乾燥 〜 1：飽和）", {})],
    [("g_w", {"bold": True, "color": DEEP}), ("：散乱の向き．濡れると前方に偏り，光が深く入る", {})]],
    size=21, spacing=10)
rect(s, 0.83, 7.9, 18.33, 1.5, CARD)
text(s, 1.17, 8.1, 17.6, 1.2, [
    [("ポイント：", {"bold": True, "color": TITLE}),
     ("濡れによる見え方の変化は，その材質の色（a）によって形が変わる．", {}),
     ("製品ごとに応答が違うのは物理的に自然", {"bold": True, "color": DEEP})]], size=24)
text(s, 0.83, 9.7, 18.33, 0.8,
     "M. Shimano, H. Okawa, Y. Asano, R. Bise, K. Nishino, I. Sato, “Wetness and Color from a Single Multispectral Image,” CVPR 2017（式 3〜7）",
     size=18, color=MUTED)

# ------------------------------------------------------ 12 our model ---
s = new_slide("ソフトカプセル向けのモデル式",
              "論文の式を，カプセルの集まり・入れ子のフィルタ・工程の変化に合わせて拡張", 13)
picture(s, "eq_model.png", 0.83, 2.35, w=9.4)
cols = [(10.85, 0.9), (11.85, 7.3)]
table(s, 2.35, cols, ["式", "意味"], [
    (["(1)", "散乱回数を平均 n̄ のポアソン分布とみなすと，論文の和が閉じた形になる（j：小帯域）"], 1.0),
    (["(2)", "濡れると散乱回数が増える（論文の g_w に対応）"], 1.0),
    (["(3)", "乾燥で皮膜が縮み，色素の効き方が変わる（κ）"], 1.0),
    (["(4)", "含水率 m から表面の濡れ γ へ．m* より乾くと表面が乾く（工程の段階）"], 1.0),
    (["(5)", "各フィルタは入れ子：長い波長の小帯域を光量の割合 s で足し合わせる"], 1.0),
], size=20)
text(s, 0.83, 8.25, 18.33, 1.7, [
    "油は 860 nm 以上で水と同じように見えるため，水の項に含めて扱う（分けない）",
    "比べるときは，共通の明るさ（光源・カプセルの写り込み量）と波長ごとの固定の差を除く",
    "係数は製品ごとに 10 個（n̄_d，n̄_s/n̄_d，r₀，u⁰ の 4 小帯域，κ，m*，w）．v_j（水の吸収）と s（光量の割合）は既知"],
    size=20, color=BODY, spacing=6)

# ------------------------------------------------ 13 model behaviour ---
s = new_slide("モデルの各項の効き方（説明用の係数）",
              "濡れと色素の項が，含水率に対する応答を曲げ，製品ごとに形を変える", 14)
picture(s, "fig_model_concept.png", 0.83, 2.3, w=18.33)
for x, head, body in ((0.83, "① 水の吸収だけ", "直線的で，変化は小さい（数 %）"),
                      (7.07, "② 濡れて散乱が増える", "光路が長くなり大きく曲がる．m* より乾くと変化が止まる"),
                      (13.31, "③ 色素の変化", "短い波長ほど効き，応答の形が変わる．製品の色によって違う")):
    rect(s, x, 8.55, 5.86, 1.75, CARD)
    text(s, x + 0.3, 8.7, 5.3, 0.45, head, size=21, bold=True, color=TITLE)
    text(s, x + 0.3, 9.2, 5.3, 1.0, body, size=19)

# ------------------------------------------------ 14 inversion & fit ---
s = new_slide("推定の方法と，当てはめの現状",
              "観測から含水率を逆算する．係数の一部はまだ決まり切っていない", 15)
boxes = [("観測", "4 つのフィルタの明るさ（15 分ごと）"),
         ("順モデル", "式 (1)〜(5)，製品ごとの係数"),
         ("逆算", "含水率を細かく区切って比べる．含水率は増えない制約"),
         ("推定値", "含水率の時系列")]
for k, (head, body) in enumerate(boxes):
    x = 0.83 + k * 4.75
    rect(s, x, 2.35, 3.95, 1.9, CARD)
    text(s, x + 0.25, 2.5, 3.5, 0.45, head, size=21, bold=True, color=TITLE)
    text(s, x + 0.25, 3.0, 3.5, 1.2, body, size=19)
    if k < 3:
        arrow(s, x + 4.03, 3.05, 0.62, 0.5)
cols = [(1.04, 4.4), (5.6, 1.7), (7.4, 1.8)]
table(s, 4.7, cols, ["係数（抜粋）", "しずく", "赤"], [
    (["濡れによる散乱の増え方 n̄_s/n̄_d", "1.38", "1.14"], 0.9),
    (["色素の効き方の変化 κ", "−0.089", "+0.064 / −0.123"], 0.9),
    (["段階の切り替わり m*", "23.0 %", "18.7 / 26.8 %"], 0.9),
], size=19)
text(s, 0.83, 8.15, 8.6, 1.6,
     "赤は当てはめの初期値の選び方で κ・m* が大きく変わる（2 回の結果を併記）．一部の係数は探索範囲の端にあり，今のデータでは決まり切っていない",
     size=18, color=MUTED)
cols = [(10.25, 5.0), (15.35, 1.7), (17.15, 1.9)]
table(s, 4.7, cols, ["誤差（%ポイント）", "しずく", "赤"], [
    (["画像による推定", "1.12", "2.41"], 0.9),
    (["時間の順番をばらばらにした対照", "1.5〜29", "2.2〜14"], 0.9),
    (["経過時間だけ（直線）", "1.25", "1.44"], 0.9),
], size=19)
text(s, 10.04, 8.15, 9.1, 1.6,
     "しずくは対照より明確に良く，画像に含水率の情報がある．赤は対照と同程度で，今の近赤外の波長では情報が取れていない",
     size=18, color=MUTED)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
