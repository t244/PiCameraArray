"""Acoustic-image fusion plan for the acoustic team, in the 9/4 deck's style.

Reuses the helpers of build_1112.py (everything above its first slide)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "build_1112.py").read_text()
helpers = src.split("# ---------------------------------------------------------------- 1 cover ---")[0]
exec(compile(helpers, "build_1112_helpers", "exec"))      # prs, text, rect, table, card, picture, new_slide ...

from pptx.enum.shapes import MSO_CONNECTOR  # noqa: E402


def arrow_right(slide, x, y, w, h, color=TITLE):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = rgb(color); a.line.fill.background(); a.shadow.inherit = False


def line(slide, x0, y0, x1, y1, color=MUTED, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = rgb(color); c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle"})
    ln.append(tail)
    return c


# ---------------------------------------------------------------- 1 cover ---
s = prs.slides.add_slide(LAYOUT)
s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(WHITE)
text(s, -0.30, 3.37, 20.60, 2.24,
     [[("音響と画像の含水率推定", {"size": 66})], [("統合の方針（案）", {"size": 60})]],
     bold=True, color=TITLE, align=PP_ALIGN.CENTER)
rect(s, 5.26, 5.77, 9.48, 0.12, GOLD)
text(s, 1.75, 6.46, 16.50, 1.04, "髙谷 剛志", size=48, bold=True, color="000000", align=PP_ALIGN.CENTER)
text(s, 1.75, 7.54, 16.50, 0.69, "筑波大学システム情報系・助教", size=36, color="000000", align=PP_ALIGN.CENTER)
text(s, 1.75, 8.30, 16.50, 0.69, "2026/11/06", size=36, color="000000", align=PP_ALIGN.CENTER)
text(s, 19.16, 10.58, 0.6, 0.36, "1", size=18, color=PAGE, align=PP_ALIGN.RIGHT)

# ------------------------------------------------------------ 2 options ---
s = new_slide("統合の方法の比較", "最終形は状態空間モデル．まずは推定値どうしの重み付き平均から始める", 2)
cols = [(1.04, 4.2), (5.45, 7.0), (12.65, 6.4)]
table(s, 2.4, cols, ["方法", "内容", "今回の向き不向き"], [
    (["① 特徴量を並べて回帰", "音響の特徴量に画像の特徴量を足し，1 つの回帰に入れる",
      "不向き．正解が 1 ロットあたり 10 点前後しかなく，特徴量を増やすと過学習しやすい．画像の出力は物理モデルの逆算結果で，音響特徴量と性質が違う"], 1.75),
    (["② 推定値の重み付き平均", "各推定器が含水率の推定値と不確かさを出し，不確かさの小さい方を重くして平均",
      "第 1 段階に向く．簡単で，片方が効かない場合（今の赤の画像）は重みが自動で小さくなる"], 1.45),
    (["③ 状態空間モデル", "含水率を隠れた時系列とし，乾燥の進み方（増えない）と，音響・画像それぞれの観測モデルで同時に推定",
      "最終形に向く．ロットごとの音響のずれを推定でき，苦手な区間を補い合える（次のページ）"], 1.45),
])
text(s, 0.83, 7.9, 18.33, 1.5, [
    "大綱 7 章の見積もり（画像だけで約 0.14 が要る）は②の考え方で，2 つの誤差が独立であることが前提",
    "実際は両方とも経過時間やロットの影響を受けて誤差が相関しうるため，統合の効果は同時計測のデータで実測して確かめる"],
    size=21, color=BODY, spacing=8)

# ---------------------------------------------------------- 3 stages ---
s = new_slide("段階的な進め方", "第 1 段階で比較の土台を作り，第 2 段階で状態空間モデルへ", 3)
rect(s, 0.83, 2.35, 8.0, 3.6, CARD)
text(s, 1.17, 2.55, 7.4, 0.45, "第 1 段階（1 月まで：2 月の比較に間に合わせる）", size=21, bold=True, color=TITLE)
text(s, 1.17, 3.1, 7.4, 2.8, [
    "出力の形をそろえる（次のページ）",
    "推定値に不確かさを付け，ロットを抜く評価（LOGO）で実際の誤差に合わせる",
    "重み付き平均を，経過時間だけ・音響だけ・画像だけと同じ分け方で比べる．重みは学習側のロットだけで決める"],
    size=20, spacing=6)
rect(s, 0.83, 6.15, 8.0, 3.9, CARD)
text(s, 1.17, 6.35, 7.4, 0.45, "第 2 段階（2〜3 月：状態空間モデル）", size=21, bold=True, color=TITLE)
text(s, 1.17, 6.9, 7.4, 3.1, [
    "ロットごとの音響のずれを，画像が押さえる時系列の形から推定できる",
    "吸油布の後（規格の境目付近）は画像が見えにくいが音は影響を受けない．前半はしずくで画像がよく効く",
    "経過時間は乾燥の進み方に含まれ，「規格に入った確率」で連続的に判定できる（止めるのが早すぎる／遅すぎるの評価）"],
    size=20, spacing=6)
# state-space diagram on the right
X0, Y = 9.6, 5.05
text(s, X0, 2.35, 9.6, 0.45, "状態空間モデルの構成", size=21, bold=True, color=TITLE)
for k, lab in enumerate(["m(t₁)", "m(t₂)", "m(t₃)"]):
    x = X0 + 0.2 + k * 3.25
    rect(s, x, Y, 2.3, 1.0, DEEP)
    text(s, x, Y + 0.22, 2.3, 0.55, lab, size=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if k < 2:
        arrow_right(s, x + 2.4, Y + 0.3, 0.75, 0.4)
    line(s, x + 1.15, Y, x + 1.15, Y - 1.05, color="11709C")
    line(s, x + 1.15, Y + 1.0, x + 1.15, Y + 2.05, color="C3592C")
rect(s, X0 + 0.2, Y - 2.2, 8.8, 1.15, "E6F0F5")
text(s, X0 + 0.45, Y - 2.15, 8.4, 1.05, [
    [("画像の観測", {"bold": True, "color": "11709C"})],
    [("x(t) ＝ Φ( m(t) ; 製品の係数 ) ＋ ノイズ　（物理順モデル）", {})]], size=19)
rect(s, X0 + 0.2, Y + 2.15, 8.8, 1.25, "F7EAE3")
text(s, X0 + 0.45, Y + 2.25, 8.4, 1.1, [
    [("音響の観測", {"bold": True, "color": "C3592C"})],
    [("f(t) ＝ g( m(t) ) ＋ ロットごとのずれ b ＋ ノイズ", {})]], size=19)
text(s, X0 + 0.2, Y + 3.5, 9.2, 0.45, "m(t)：含水率（隠れた時系列）．乾燥中は増えない", size=19, bold=True, color=DEEP)
text(s, X0 + 0.2, Y + 4.0, 9.2, 0.8, "観測の間隔が違っても（画像 2 分，音 1 分）そのまま扱える．片方が欠けた区間も推定が続く",
     size=18, color=MUTED)

# ------------------------------------------------------- 4 interface ---
s = new_slide("出力の形の取り決め（案）", "推定器ごとに 1 つの CSV．採取点の時刻だけでなく，時系列で出す", 4)
cols = [(1.04, 3.3), (4.55, 2.4), (7.15, 11.9)]
table(s, 2.35, cols, ["列名", "単位", "意味"], [
    (["lot_id", "−", "ロット番号（台帳と共通）"], 0.62),
    (["time", "実時刻", "日本時間，ISO 8601（例：2026-12-03T14:02:00+09:00）．時計合わせ後の時刻"], 0.62),
    (["elapsed_min", "分", "乾燥開始からの経過時間"], 0.62),
    (["sample_id", "−", "その時刻が採取点なら台帳の番号，それ以外は空"], 0.62),
    (["moisture_est", "%", "含水率の推定値"], 0.62),
    (["moisture_se", "%ポイント", "推定値の標準誤差（ロットを抜く評価で実際の誤差に合わせた値）"], 0.62),
    (["valid", "0 / 1", "使えるかどうか（吸油布・欠測・機器の異常など．理由は reason 列に）"], 0.62),
    (["model_id", "−", "推定器の名前と版（例：acoustic_M3_v2，image_physics_v1）"], 0.62),
], size=19)
text(s, 0.83, 8.0, 18.33, 2.0, [
    "時間の間隔：音響は 1 分ごと，画像は 2 分ごと（各推定器の自然な間隔のまま．そろえるのは統合側で行う）",
    "第 2 段階に向けて，音響特徴量そのものも同じ lot_id・time で 1 分ごとの時系列として出していただきたい",
    "評価の分け方（どのロットを抜くか）と，推定値に使ったロットの一覧も一緒に記録する"],
    size=20, color=BODY, spacing=6)

# ------------------------------------------------------- 5 requests ---
s = new_slide("お願いと注意点", "同時計測のデータで，同じ条件・同じ物差しで比べられるようにする", 5)
card(s, 0.83, 2.4, 8.9, 3.9, "音響チームへのお願い", "3 点",
     "① 推定値に不確かさ（標準誤差）を付ける　② 特徴量を 1 分ごとの時系列でも出す　③ 台帳と時計合わせで時刻をそろえる", big_size=40)
card(s, 10.27, 2.4, 8.9, 3.9, "画像側でやること", "2 点",
     "① 画像の推定式の不具合を直して評価し直す　② 同じ形で推定値と不確かさを出す", big_size=40)
text(s, 0.83, 6.7, 18.33, 3.2, [
    "評価：ロット単位の入れ子の LOGO．重みや係数の選択は学習側のロットの中だけで行う",
    "比べる相手：経過時間だけ（M0）・音響だけ・画像だけ・統合．全体とタンブラー終盤（規格の境目付近）を分けて報告",
    "今の画像は，赤では近赤外の情報がない．統合の重みが自然に小さくなることも確認する",
    "統合で精度が上がらない場合も，その理由（誤差の相関・ロットのずれ）を示すこと自体を成果とする"],
    size=21, color=BODY, spacing=8)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
