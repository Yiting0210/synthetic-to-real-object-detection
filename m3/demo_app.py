"""M3 live demo: a local web page with the two required modes.

    python demo_app.py                      # opens http://127.0.0.1:7860
    python demo_app.py --models-dir ../models --examples examples

Tab 1 "Single object → ID": the M1 classifier (MobileNetV2) gives the object ID and its top-3 probabilities.
Tab 2 "Multi-object → IDs + locations": the M2 detector (YOLOv8n) draws every object with its ID and lists the boxes.
Example images (held-out test images exported by the M2 notebook) are listed under each tab; clicking one runs the model.
Inference lives in predict.py; this file only builds the page.
"""
import argparse
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from predict import Classifier, Detector, draw

ap = argparse.ArgumentParser()
ap.add_argument("--models-dir", type=Path, default=Path(__file__).parent / "models")
ap.add_argument("--examples", type=Path, default=Path(__file__).parent / "examples")
ap.add_argument("--port", type=int, default=7860)
args = ap.parse_args()

classifier = Classifier(args.models_dir) if (args.models_dir / "mobilenetv2.keras").exists() else None
detector = Detector(args.models_dir) if (args.models_dir / "d1_yolov8n_best.pt").exists() else None

# Warm-up: the first prediction is slow (model initialization), so do it before the demo starts
_blank = Image.new("RGB", (224, 224), (114, 114, 114))
if classifier:
    classifier(_blank)
if detector:
    detector(_blank)

# Test results reported in M1 and M2 (shown on the page; not recomputed here)
M1_RESULT = "test accuracy 0.993"
M2_RESULT = "D1-test mAP@0.5 0.928"   # retrained detector (m2/models); the submitted M2 model had 0.908
REPO_URL = "https://github.com/Yiting0210/synthetic-to-real-object-detection"   # header GitHub icon; inactive if empty

# ---------------------------------------------------------------- icons (inline SVG, no icon library needed)
def _svg(body, size=20, sw=1.8):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>')

IC = {
    "cube": '<path d="M12 2 3 7v10l9 5 9-5V7l-9-5z"/><path d="M3 7l9 5 9-5"/><path d="M12 12v10"/>',
    "home": '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
    "chart": '<path d="M4 20h16"/><rect x="6" y="11" width="3" height="7" rx="1"/><rect x="11" y="7" width="3" height="11" rx="1"/><rect x="16" y="4" width="3" height="14" rx="1"/>',
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "github": '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9a3.4 3.4 0 0 0-.9-2.6c3.1-.4 6.4-1.5 6.4-6.9A5.4 5.4 0 0 0 20 4.8 5 5 0 0 0 19.9 1S18.7.6 16 2.5a13.4 13.4 0 0 0-7 0C6.3.6 5.1 1 5.1 1A5 5 0 0 0 5 4.8a5.4 5.4 0 0 0-1.5 3.8c0 5.4 3.3 6.5 6.4 6.9a3.4 3.4 0 0 0-.9 2.6V22"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/>',
    "frame": '<rect x="5" y="5" width="14" height="14" rx="1.5"/><rect x="2.5" y="2.5" width="5" height="5" rx="1.2"/><rect x="16.5" y="2.5" width="5" height="5" rx="1.2"/><rect x="2.5" y="16.5" width="5" height="5" rx="1.2"/><rect x="16.5" y="16.5" width="5" height="5" rx="1.2"/>',
    "arrow": '<path d="M9 6l6 6-6 6"/>',
    "upload": '<path d="M12 16V4"/><path d="M7 9l5-5 5 5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13.5" r="3.5"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="9" cy="10" r="1.8"/><path d="M21 16l-5-5-9 9"/>',
    "trash": '<path d="M4 7h16"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M6 7l1 13h10l1-13"/><path d="M9 7V4h6v3"/>',
    "code": '<path d="M8 8l-4 4 4 4"/><path d="M16 8l4 4-4 4"/><path d="M13.5 5l-3 14"/>',
    "list": '<path d="M9 6h11"/><path d="M9 12h11"/><path d="M9 18h11"/><circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/>',
}

CUBE_ART = """<svg class="deco-cube" viewBox="0 0 120 120" aria-hidden="true">
  <ellipse cx="60" cy="104" rx="40" ry="8" fill="#A78BFA" opacity=".18"/>
  <path d="M60 16 96 36v42L60 98 24 78V36z" fill="#C7D2FE" opacity=".7"/>
  <path d="M60 56 96 36v42L60 98z" fill="#A78BFA" opacity=".55"/>
  <path d="M60 16 96 36 60 56 24 36z" fill="#F5F3FF" opacity=".95"/>
</svg>"""   # flat fills: gradient ids would clash between the two tabs

# ---------------------------------------------------------------- HTML pieces
def header_html():
    gh = (f'<a class="nav-item icon-only" href="{REPO_URL}" target="_blank" rel="noopener" title="GitHub repository">{_svg(IC["github"])}</a>'
          if REPO_URL else f'<span class="nav-item icon-only disabled" title="Repository link not set yet">{_svg(IC["github"])}</span>')
    return f"""
<div id="top" class="page-header">
  <div class="brand">
    <div class="logo">{_svg(IC["cube"], 28, 1.9)}</div>
    <div>
      <div class="title">Lab-to-Real Object ID</div>
      <div class="subtitle">IE 7615 · 20 object classes · Identify a single object or identify and locate every object in a multi-object image</div>
    </div>
  </div>
  <nav class="nav">
    <button class="nav-item active" data-act="scroll" data-target="#top">{_svg(IC["home"], 18)}<span>Home</span></button>
    <button class="nav-item" data-act="scroll" data-target=".right-col">{_svg(IC["chart"], 18)}<span>Results</span></button>
    <button class="nav-item" data-act="scroll" data-target=".ex-card">{_svg(IC["db"], 18)}<span>Examples</span></button>
    {gh}
  </nav>
</div>
<div class="model-cards">
  <button class="model-card mc-1" data-act="tab" data-idx="0">
    <span class="mc-icon">{_svg(IC["target"], 34, 1.8)}</span>
    <span class="mc-text"><span class="mc-title">Single object → ID</span>
      <span class="mc-sub">M1 classifier · MobileNetV2 · {M1_RESULT}</span></span>
    <span class="mc-arrow">{_svg(IC["arrow"], 20, 2.2)}</span>
  </button>
  <button class="model-card mc-2" data-act="tab" data-idx="1">
    <span class="mc-icon">{_svg(IC["frame"], 34, 1.6)}</span>
    <span class="mc-text"><span class="mc-title">Multi-object → IDs + locations</span>
      <span class="mc-sub">M2 detector · YOLOv8n · {M2_RESULT}</span></span>
    <span class="mc-arrow">{_svg(IC["arrow"], 20, 2.2)}</span>
  </button>
</div>"""


def toolbar_html(img_id, examples_id):
    items = [("upload", "Upload", "upload", ""), ("camera", "Camera", "camera", ""),
             ("examples", "Examples", "image", f' data-target="#{examples_id}"'), ("clear", "Clear", "trash", "")]
    btns = "".join(f'<button class="tb" data-act="{a}"{extra}>{_svg(IC[ic], 22)}<span>{label}</span></button>'
                   for a, label, ic, extra in items)
    return f'<div class="toolbar" data-for="{img_id}">{btns}</div>'


def card_head(icon, title, sub=""):
    sub = f'<span class="ch-sub">{sub}</span>' if sub else ""
    return f'<div class="card-head">{_svg(IC[icon], 18)}<span class="ch-title">{title}</span>{sub}</div>'


def summary_html(label, value, meta, chips=None):
    chip_html = ""
    if chips is not None:
        chip_html = '<div class="chips">' + ("".join(f'<span class="chip">{c}</span>' for c in chips)
                                             or '<span class="muted">none above the threshold</span>') + "</div>"
    return (f'<div class="summary-card"><div class="sc-label">{label}</div><div class="sc-value">{value}</div>'
            f'<div class="sc-meta">{meta}</div>{chip_html}{CUBE_ART}</div>')


def pct(p, decimals):
    """Percentage text that never rounds up to 100% or down to a misleading 0%."""
    hi, lo = 1 - 0.5 * 10 ** -(decimals + 2), 0.5 * 10 ** -(decimals + 2)   # e.g. 0.9995 / 0.0005 for 1 decimal
    if p >= hi:
        return f"> {100 - 10 ** -decimals:.{decimals}f}%"     # "> 99.9%" or "> 99%"
    if 0 < p < lo:
        return f"< {10 ** -decimals:.{decimals}f}%"           # "< 0.1%" or "< 1%"
    return f"{p * 100:.{decimals}f}%"


def prob_html(top_k):
    rows = []
    for k, o in enumerate(top_k):
        p = o["probability"]
        width = "0" if p < 0.005 else f"max(6px, {p * 100:.2f}%)"          # 0% stays empty; small values stay visible
        rows.append(f'<div class="pr-row"><span class="pr-label">{o["object_id"]}</span>'
                    f'<span class="pr-track"><span class="pr-fill {"top" if k == 0 else "other"}" style="width:{width}"></span></span>'
                    f'<span class="pr-pct">{pct(p, 0)}</span></div>')
    body = "".join(rows) if rows else '<div class="muted">The three most likely IDs appear here.</div>'
    return f'<div class="prob-card">{card_head("chart", "Top-3 probabilities")}{body}</div>'


EMPTY_SINGLE = summary_html("Predicted object ID", "—", "Upload an image, use the camera, or click an example below.")
EMPTY_MULTI = summary_html("Detected objects", "—", "Upload a multi-object image or click an example below.", chips=[])
EMPTY_PROB = prob_html([])


def examples(sub):
    d = args.examples / sub
    return [[str(p)] for p in sorted(d.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if d.exists() else []


# ---------------------------------------------------------------- callbacks (inference via predict.py, unchanged)
def run_single(image):
    if image is None:
        return EMPTY_SINGLE, EMPTY_PROB, {}
    t = time.perf_counter()
    out = classifier(image.convert("RGB"))
    ms = (time.perf_counter() - t) * 1000
    card = summary_html("Predicted object ID", out["object_id"], f"Confidence {pct(out['confidence'], 1)} · MobileNetV2 · {ms:.0f} ms")
    return card, prob_html(out["top_k"]), out


def run_multi(image, conf):
    if image is None:
        return None, EMPTY_MULTI, [], {}
    image = image.convert("RGB")
    t = time.perf_counter()
    out = detector(image, conf=conf)
    ms = (time.perf_counter() - t) * 1000
    objs = out["objects"]
    card = summary_html("Detected objects", f"{len(objs)} object{'s' if len(objs) != 1 else ''}",
                        f"Threshold {conf:.2f} · YOLOv8n · {ms:.0f} ms", chips=[o["object_id"] for o in objs])
    rows = [[k + 1, o["object_id"], round(o["confidence"], 3), *[round(v) for v in o["box_xyxy"]]] for k, o in enumerate(objs)]
    return draw(image, objs, detector.classes), card, rows, out


# ---------------------------------------------------------------- page behavior (toolbar, nav, model cards)
JS = """
() => {
  const visible = (sel) => [...document.querySelectorAll(sel)].find(el => el.offsetParent !== null);
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const act = b.dataset.act;
    if (act === 'tab') { const tabs = [...document.querySelectorAll('#task-tabs [role=tab]')]; tabs[+b.dataset.idx]?.click(); return; }
    if (act === 'scroll' || act === 'examples') { visible(b.dataset.target)?.scrollIntoView({behavior: 'smooth', block: act === 'examples' ? 'center' : 'start'}); return; }
    const host = b.closest('[data-for]');
    const img = host && document.getElementById(host.dataset.for);
    if (!img) return;
    if (act === 'upload') {
      img.querySelector('button[aria-label="Upload file"]')?.click();
      setTimeout(() => img.querySelector('input[type=file]')?.click(), 60);
    } else if (act === 'camera') {
      img.querySelector('button[aria-label="Capture from camera"]')?.click();
    } else if (act === 'clear') {
      const c = [...img.querySelectorAll('button')].find(x => /clear|remove/i.test((x.getAttribute('aria-label') || '') + (x.title || '')));
      c?.click();
    }
  });
  document.addEventListener('click', (e) => {              // highlight the clicked example thumbnail
    const g = e.target.closest('.gallery-item');
    if (!g) return;
    g.parentElement.querySelectorAll('.gallery-item').forEach(x => x.classList.toggle('is-selected', x === g));
  }, true);
}
"""

# ---------------------------------------------------------------- styles
CSS = """
:root {
  --bg: #F7F8FC; --surface: #FFFFFF; --surface-soft: #F5F3FF;
  --text-primary: #172554; --text-secondary: #64748B; --text-muted: #94A3B8;
  --border: #E6EAF2; --border-strong: #D8DEEA;
  --blue: #4F6EF7; --blue-light: #6F8BFF; --purple: #7C3AED; --purple-light: #9B6BFF;
  --radius-card: 18px; --radius-inner: 14px; --radius-button: 11px; --radius-small: 9px;
  --shadow-card: 0 8px 30px rgba(30, 41, 59, 0.055); --shadow-hover: 0 12px 36px rgba(30, 41, 59, 0.08);
  --ease: color 160ms ease, background-color 160ms ease, border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}
body, gradio-app, .gradio-container, .main, .wrap.app {
  background: radial-gradient(circle at 90% 0%, rgba(124,58,237,.08), transparent 28%),
              radial-gradient(circle at 8% 15%, rgba(79,110,247,.08), transparent 24%), #F7F8FC !important;
}
.gradio-container { width: 100% !important; max-width: 1440px !important; box-sizing: border-box; margin: 0 auto !important; padding: 20px 32px 32px !important;
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; color: var(--text-primary); }
.gradio-container .main, .gradio-container .contain, .gradio-container main.fillable { padding: 0 !important; }
footer { display: none !important; }
button, a, .gallery-item { transition: var(--ease); }
.muted { color: var(--text-muted); font-size: 14px; }

/* blocks: no default chrome; cards are styled explicitly */
.gradio-container .block { border: 0 !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; }
.gradio-container .form { border: 0 !important; background: transparent !important; box-shadow: none !important; }
.html-container, .prose { padding: 0 !important; }

/* 4. header */
.page-header { height: 82px; display: flex; align-items: center; justify-content: space-between; gap: 24px; }
.brand { display: flex; align-items: center; gap: 16px; min-width: 0; }
.logo { width: 48px; height: 48px; flex: none; border-radius: 12px; display: grid; place-items: center; color: #fff;
  background: linear-gradient(135deg, #5B8DEF 0%, #5865E8 48%, #7C3AED 100%); box-shadow: 0 6px 16px rgba(88,101,232,.25); }
.title { font-size: 32px; line-height: 1.1; font-weight: 750; letter-spacing: -0.6px; color: #172554; }
.subtitle { margin-top: 6px; font-size: 14px; font-weight: 400; color: #64748B; }
.nav { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
.nav-item { height: 42px; padding: 0 14px; border-radius: 12px; display: inline-flex; align-items: center; gap: 8px;
  background: transparent; border: 0; color: #24324A; font-size: 14px; font-weight: 500; cursor: pointer; text-decoration: none; }
.nav-item:hover { background: rgba(79,110,247,.06); color: #4F46E5; }
.nav-item.active { color: #4F46E5; }
.nav-item.icon-only { width: 42px; padding: 0; justify-content: center; }
.nav-item.disabled { color: var(--text-muted); cursor: default; }
.nav-item.disabled:hover { background: transparent; color: var(--text-muted); }

/* 5. model cards */
.model-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }
.model-card { height: 100px; border-radius: 18px; padding: 0 28px; display: flex; align-items: center; gap: 22px;
  border: 0; cursor: pointer; text-align: left; color: #fff; }
.model-card:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(79,70,229,.16); }
.mc-1 { background: linear-gradient(135deg, #5B8DEF 0%, #3F51C7 100%); }
.mc-2 { background: linear-gradient(135deg, #9A67F2 0%, #6D35E5 100%); }
.mc-icon { flex: none; width: 54px; height: 54px; border-radius: 50%; display: grid; place-items: center;
  background: rgba(255,255,255,.12); color: rgba(255,255,255,.92); }
.mc-text { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.mc-title { font-size: 19px; font-weight: 700; color: white; }
.mc-sub { font-size: 14px; color: rgba(255,255,255,.88); margin-top: 5px; }
.mc-arrow { flex: none; width: 42px; height: 42px; border-radius: 50%; display: grid; place-items: center;
  background: rgba(255,255,255,.10); color: #fff; }
.model-card:hover .mc-arrow { background: rgba(255,255,255,.18); }

/* 6. tabs */
#task-tabs { margin-top: 8px; gap: 0 !important; }
#task-tabs .tabitem, #task-tabs .tabitem > div { gap: 0 !important; }
#task-tabs > .tab-wrapper { height: 66px; display: flex; align-items: flex-end; border-bottom: 1px solid #E6EAF2; padding: 0; }
#task-tabs .tab-container { display: flex; gap: 34px; height: 56px; border: 0 !important; }
#task-tabs .tab-container::after, #task-tabs .tab-wrapper::after { display: none !important; }
#task-tabs button[role=tab] { height: 56px; display: flex; align-items: center; gap: 9px; position: relative; padding: 0 2px !important;
  font-size: 15px; color: #24324A; font-weight: 500; background: transparent !important; border: 0 !important; margin: 0 !important; }
#task-tabs button[role=tab]::before { content: ""; width: 20px; height: 20px; background: currentColor; flex: none;
  -webkit-mask: var(--tab-icon) center / contain no-repeat; mask: var(--tab-icon) center / contain no-repeat; }
#task-tabs button[role=tab]:nth-child(1) { --tab-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linejoin='round'%3E%3Cpath d='M12 2 3 7v10l9 5 9-5V7l-9-5z'/%3E%3Cpath d='M3 7l9 5 9-5M12 12v10'/%3E%3C/svg%3E"); }
#task-tabs button[role=tab]:nth-child(2) { --tab-icon: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.6'%3E%3Crect x='5' y='5' width='14' height='14' rx='1.5'/%3E%3Crect x='2.5' y='2.5' width='5' height='5' rx='1.2'/%3E%3Crect x='16.5' y='2.5' width='5' height='5' rx='1.2'/%3E%3Crect x='2.5' y='16.5' width='5' height='5' rx='1.2'/%3E%3Crect x='16.5' y='16.5' width='5' height='5' rx='1.2'/%3E%3C/svg%3E"); }
#task-tabs button[role=tab]:hover { color: #4F46E5; }
#task-tabs button[role=tab].selected { color: #4F46E5; font-weight: 650; }
#task-tabs button[role=tab].selected::after { content: ""; position: absolute; height: 2px; bottom: -1px; left: 0; right: 0;
  background: #6366F1; border-radius: 2px; width: auto !important; }
#task-tabs .tabitem { padding: 0 !important; border: 0 !important; background: transparent !important; }

/* 7. workspace */
.workspace { display: grid !important; grid-template-columns: minmax(0, 1.03fr) minmax(0, 0.97fr); gap: 24px !important;
  margin-top: 28px; align-items: start !important; }
.workspace > .column { min-width: 0 !important; width: auto !important; }

/* 8-11. left image card */
.img-card { background: #FFFFFF; border: 1px solid #E6EAF2; border-radius: 18px; box-shadow: var(--shadow-card);
  overflow: hidden; gap: 0 !important; }
.img-card .block { position: relative; }
.img-card .block > label.float, .img-card .block > label { position: absolute !important; top: 0; left: 0; height: 58px; padding: 0 20px !important;
  display: flex; align-items: center; gap: 10px; background: transparent !important; border: 0 !important; box-shadow: none !important;
  font-size: 16px !important; font-weight: 650; color: #172554 !important; z-index: 2; margin: 0 !important; }
.img-card .block > label svg { width: 18px; height: 18px; color: #64748B; }
.img-card .image-container { padding-top: 58px; height: auto !important; }
.img-card .image-container .top-panel { top: 12px !important; right: 16px !important; gap: 4px; background: transparent !important;
  border: 0 !important; box-shadow: none !important; }
.img-card .top-panel button { width: 34px !important; height: 34px !important; border-radius: 9px !important; border: 0 !important;
  background: transparent !important; color: #64748B !important; }
.img-card .top-panel button:hover { background: #F5F3FF !important; color: #4F46E5 !important; }
.img-card .source-selection { display: none !important; }
.img-card .upload-container, .img-card .image-frame, .img-card .webcam, .img-card .image-container .wrap { min-height: 350px; }   /* not the slider's .wrap */
.img-card .upload-container { margin: 0 20px; border-radius: 14px; background: #FAFBFE; border: 1px dashed #D8DEEA; }
.img-card .upload-container:hover { border-color: #A5B4FC; background: #F5F3FF; }
.img-card .upload-container:has(.image-frame), .img-card .upload-container:has(.image-frame):hover { border-color: transparent; background: transparent; }
.img-card .image-frame { display: flex !important; align-items: center; justify-content: center; height: 350px !important; }
.img-card .image-frame img { width: 300px !important; height: 300px !important; max-width: 320px !important; max-height: 310px !important;
  object-fit: contain; border-radius: 14px; }
.output-img .image-frame img { max-width: 100% !important; max-height: 420px !important; }
.toolbar-block { overflow: visible !important; }
.toolbar { display: flex; justify-content: center; gap: 54px; height: 70px; align-items: center; }
.tb { display: flex; flex-direction: column; align-items: center; gap: 5px; font-size: 12.5px; color: #64748B; background: transparent;
  border: 0; cursor: pointer; padding: 4px 6px; border-radius: 9px; }
.tb:hover, .tb.active { color: #4F46E5; }
.btn-wrap { padding: 0 22px 22px !important; }
.primary-btn { width: 100%; height: 50px !important; max-height: 52px; border: 0 !important; border-radius: 11px !important;
  background: linear-gradient(90deg, #696BF3 0%, #7C3AED 100%) !important; color: white !important; font-size: 16px !important;
  font-weight: 700 !important; box-shadow: 0 8px 18px rgba(99,102,241,.18) !important; }
.primary-btn:hover { transform: translateY(-1px); box-shadow: 0 10px 22px rgba(99,102,241,.23) !important; }
.slider-wrap { padding: 0 22px 14px !important; }
.slider-wrap .block { background: transparent !important; }
.slider-wrap label span, .slider-wrap .head { color: #172554 !important; font-weight: 600; font-size: 14px; }

/* 12. right column */
.right-col { display: flex !important; flex-direction: column !important; gap: 18px !important; }

/* 13. prediction summary */
.summary-card { min-height: 180px; border: 1px solid #DDE3F0; border-radius: 18px; padding: 28px 40px; position: relative; overflow: hidden;
  background: linear-gradient(120deg, #FFFFFF 0%, #F5F3FF 60%, #EEF2FF 100%); box-sizing: border-box; }
.sc-label { font-size: 14px; color: #64748B; font-weight: 450; position: relative; z-index: 1; }
.sc-value { font-size: 42px; line-height: 1; font-weight: 800; margin-top: 12px; letter-spacing: -0.8px; position: relative; z-index: 1;
  background: linear-gradient(90deg, #4F46E5, #7C3AED); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; display: inline-block; }
.sc-meta { font-size: 14px; color: #64748B; margin-top: 12px; position: relative; z-index: 1; }
.deco-cube { position: absolute; right: 34px; top: 50%; transform: translateY(-50%); width: 112px; height: 112px; opacity: .85; }
.chips { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; position: relative; z-index: 1; max-width: 75%; }
.chip { background: rgba(99,102,241,.08); color: #4F46E5; border-radius: 999px; padding: 3px 10px; font-size: 13px; font-weight: 600; }

/* 14. probabilities */
.prob-card { background: white; border: 1px solid #E6EAF2; border-radius: 18px; padding: 20px 28px 24px; box-shadow: 0 8px 30px rgba(30,41,59,.04);
  min-height: 210px; box-sizing: border-box; }
.card-head { display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 700; color: #172554; }
.card-head svg { color: #6366F1; }
.prob-card .card-head { margin-bottom: 26px; }
.ch-sub { font-size: 13px; font-weight: 400; color: #64748B; margin-left: -3px; }
.pr-row { display: grid; grid-template-columns: 88px 1fr 48px; gap: 18px; align-items: center; height: 42px; }
.pr-label { font-size: 14px; font-weight: 650; color: #172554; }
.pr-pct { font-size: 14px; text-align: right; color: #334155; }
.pr-track { height: 10px; border-radius: 999px; background: #ECECF6; overflow: hidden; display: block; }
.pr-fill { display: block; height: 100%; border-radius: 999px; }
.pr-fill.top { background: linear-gradient(90deg, #A06BF3, #7140E8); }
.pr-fill.other { background: #9F8CF7; }

/* multi-object: output image and table cards */
.plain-card { background: #FFFFFF; border: 1px solid #E6EAF2; border-radius: 18px; box-shadow: var(--shadow-card); padding: 18px 20px 20px; }
.plain-card .card-head { margin-bottom: 12px; }
.plain-card .image-container { border-radius: 14px; overflow: hidden; }
.plain-card .block > label { display: none !important; }
.plain-card table { font-size: 13.5px; }
.plain-card th { background: #F8FAFC !important; color: #64748B !important; font-weight: 600 !important; }
.plain-card .table-wrap, .plain-card .table-container { border: 1px solid #E6EAF2 !important; border-radius: 12px !important; }

/* 15. JSON accordion */
.json-acc { background: white !important; border: 1px solid #E6EAF2 !important; border-radius: 14px !important; overflow: hidden; padding: 0 !important; }
.json-acc > button.label-wrap { height: 58px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px !important;
  font-size: 14px !important; font-weight: 550 !important; color: #172554 !important; margin: 0 !important; }
.json-acc > button.label-wrap > span:first-child::before { content: ""; display: inline-block; width: 18px; height: 18px; margin-right: 10px;
  vertical-align: -4px; background: #6366F1;
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Cpath d='M8 8l-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14'/%3E%3C/svg%3E") center/contain no-repeat;
  mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'%3E%3Cpath d='M8 8l-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14'/%3E%3C/svg%3E") center/contain no-repeat; }
.json-acc > div:not(.label-wrap) { background: #F8FAFC; padding: 18px 20px !important; border-top: 1px solid #E6EAF2; }
.json-acc .json-holder, .json-acc .json-node { font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important; font-size: 12.5px; line-height: 1.6; }

/* 16. examples card */
.ex-card { margin-top: 20px; background: #FFFFFF !important; border: 1px solid #E6EAF2; border-radius: 18px; padding: 16px 20px 20px !important;
  box-shadow: 0 8px 30px rgba(30,41,59,.035); gap: 0 !important; }
.ex-card .card-head { height: 38px; font-size: 15px; }
.ex-card .card-head svg { color: #64748B; }
.ex-card .block > .label, .ex-card .block > div > .label, .ex-card .label-wrap, .ex-card span[data-testid="block-label"] { display: none !important; }
.ex-card .gallery { display: flex !important; gap: 18px !important; overflow-x: auto; padding-top: 8px; flex-wrap: nowrap !important; }
.ex-card .gallery-item { padding: 0 !important; border: 0 !important; background: transparent !important; flex: none; border-radius: 10px; }
.ex-card .gallery-item .container, .ex-card .gallery-item img { width: 104px !important; height: 76px !important; max-width: none !important; max-height: none !important; }
.ex-card .gallery-item .container { border-radius: 10px !important; border: 1px solid #E6EAF2 !important; overflow: hidden; padding: 0 !important; }
.ex-card .gallery-item img { object-fit: cover !important; border-radius: 0 !important; max-width: none !important; }
.ex-card .gallery-item:hover { transform: translateY(-2px); }
.ex-card .gallery-item:hover .container { box-shadow: 0 8px 18px rgba(30,41,59,.08); }
.ex-card .gallery-item.is-selected .container { border: 2px solid #6366F1 !important; box-shadow: 0 0 0 2px rgba(99,102,241,.10); }
.ex-card .paginate { display: none; }

/* 19. responsive */
@media (max-width: 900px) {
  .workspace { grid-template-columns: 1fr; }
  .page-header { height: auto; flex-wrap: wrap; padding: 6px 0; }
  .title { font-size: 26px; }
}
@media (max-width: 720px) {
  .gradio-container { padding: 16px !important; }
  .model-cards { grid-template-columns: 1fr; }
  .model-card { height: auto; min-height: 88px; padding: 16px 18px; gap: 14px; }
  .mc-icon { width: 44px; height: 44px; }
  .mc-title { font-size: 17px; }
  .mc-arrow { width: 36px; height: 36px; }
  .toolbar { gap: 28px; }
  .summary-card { padding: 24px; }
  .deco-cube { display: none; }
}
"""

theme = gr.themes.Base(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.Font(f) for f in ("Inter", "ui-sans-serif", "Segoe UI", "sans-serif")])   # plain strings crash Gradio 5.50

with gr.Blocks(title="Lab-to-Real Object ID", theme=theme, css=CSS, js=JS, fill_width=True) as demo:
    gr.HTML(header_html())

    with gr.Tabs(elem_id="task-tabs"):
        with gr.Tab("Single object → ID"):
            if classifier is None:
                gr.Markdown(f"⚠️ `mobilenetv2.keras` not found in `{args.models_dir}`.")
            else:
                with gr.Row(elem_classes="workspace", equal_height=False):
                    with gr.Column(elem_classes="img-card", min_width=0):
                        s_in = gr.Image(type="pil", label="Single-object image", sources=["upload", "webcam"], elem_id="s-img",
                                        show_download_button=False)
                        gr.HTML(toolbar_html("s-img", "s-examples"), elem_classes="toolbar-block")
                        with gr.Column(elem_classes="btn-wrap"):
                            s_btn = gr.Button("✦ Identify", elem_classes="primary-btn")
                    with gr.Column(elem_classes="right-col", min_width=0):
                        s_card = gr.HTML(EMPTY_SINGLE)
                        s_prob = gr.HTML(EMPTY_PROB)
                        with gr.Accordion("Raw output (JSON)", open=False, elem_classes="json-acc"):
                            s_json = gr.JSON(show_label=False)
                s_outs = [s_card, s_prob, s_json]
                s_btn.click(run_single, s_in, s_outs)
                s_in.upload(run_single, s_in, s_outs)
                s_in.clear(lambda: (EMPTY_SINGLE, EMPTY_PROB, {}), None, s_outs)
                if examples("single"):
                    with gr.Column(elem_classes="ex-card", elem_id="s-examples"):
                        gr.HTML(card_head("image", "Examples", "Held-out test images"))
                        gr.Examples(examples("single"), s_in, s_outs, fn=run_single, run_on_click=True, cache_examples=False,
                                    examples_per_page=20)

        with gr.Tab("Multi-object → IDs + locations"):
            if detector is None:
                gr.Markdown(f"⚠️ `d1_yolov8n_best.pt` not found in `{args.models_dir}`.")
            else:
                with gr.Row(elem_classes="workspace", equal_height=False):
                    with gr.Column(elem_classes="img-card", min_width=0):
                        m_in = gr.Image(type="pil", label="Multi-object image", sources=["upload", "webcam"], elem_id="m-img",
                                        show_download_button=False)
                        gr.HTML(toolbar_html("m-img", "m-examples"), elem_classes="toolbar-block")
                        with gr.Column(elem_classes="slider-wrap"):
                            m_conf = gr.Slider(0.05, 0.95, value=detector.default_conf, step=0.05,
                                               label=f"Confidence threshold (chosen on validation: {detector.default_conf})")
                        with gr.Column(elem_classes="btn-wrap"):
                            m_btn = gr.Button("✦ Detect", elem_classes="primary-btn")
                    with gr.Column(elem_classes="right-col", min_width=0):
                        m_card = gr.HTML(EMPTY_MULTI)
                        with gr.Column(elem_classes="plain-card"):
                            gr.HTML(card_head("frame", "Detected objects"))
                            m_out = gr.Image(type="pil", label="Detected objects", interactive=False, elem_classes="output-img",
                                             show_download_button=True)
                        with gr.Column(elem_classes="plain-card"):
                            gr.HTML(card_head("list", "Objects", "box in pixels of the input image"))
                            m_table = gr.Dataframe(headers=["#", "Object ID", "Confidence", "x0", "y0", "x1", "y1"],
                                                   datatype=["number", "str", "number", "number", "number", "number", "number"],
                                                   interactive=False, show_label=False)
                        with gr.Accordion("Raw output (JSON)", open=False, elem_classes="json-acc"):
                            m_json = gr.JSON(show_label=False)
                m_outs = [m_out, m_card, m_table, m_json]
                m_btn.click(run_multi, [m_in, m_conf], m_outs)
                m_in.upload(run_multi, [m_in, m_conf], m_outs)
                # any change (dragging or typing in the number box) re-runs; while dragging only the last value is computed
                m_conf.change(run_multi, [m_in, m_conf], m_outs, trigger_mode="always_last")
                m_in.clear(lambda: (None, EMPTY_MULTI, [], {}), None, m_outs)
                if examples("multi"):
                    with gr.Column(elem_classes="ex-card", elem_id="m-examples"):
                        gr.HTML(card_head("image", "Examples", "Held-out test composites"))
                        gr.Examples(examples("multi"), m_in, m_outs, fn=lambda img: run_multi(img, m_conf.value),
                                    run_on_click=True, cache_examples=False, examples_per_page=20)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=args.port)
