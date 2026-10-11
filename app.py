import html
import os
import random
import re
import tempfile
import time
from pathlib import Path
 
import streamlit as st
from dotenv import load_dotenv
 
load_dotenv()
 
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.vector_store import warm_up_embeddings
from core.rag_engine import build_rag_chain, ask_question
 
# ─── Author (shown in footer) ───────────────────────────────────────────────────
AUTHOR = "Faizan"
GITHUB_URL = "https://github.com/faizan01010"
 
# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
 
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #8585b0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}
 
/* ── Global ── */
html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, .stTextInput, .stSelectbox, .stRadio {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2.4rem !important; max-width: 1200px; }
 
/* Grid + slow drifting glow behind everything */
.stApp::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}
.stApp::after {
    content: ''; position: fixed; inset: -20%; pointer-events: none; z-index: 0;
    background:
        radial-gradient(40% 35% at 20% 25%, rgba(124,58,237,0.16), transparent 70%),
        radial-gradient(35% 30% at 80% 70%, rgba(6,182,212,0.11), transparent 70%);
    animation: drift 24s ease-in-out infinite alternate;
}
@keyframes drift { to { transform: translate3d(4%, 3%, 0) scale(1.08); } }
 
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text); }
h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text); }
label, [data-testid="stWidgetLabel"] p { color: var(--text-muted) !important; font-size: 0.8rem !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }
 
/* ── Hero ── */
.hero-title {
    font-family: 'Syne', sans-serif; font-weight: 800; line-height: 1.05; margin: 0;
    font-size: clamp(2.1rem, 5vw, 3.6rem);
    background: linear-gradient(120deg, #ffffff 0%, var(--accent-glow) 35%, var(--accent-2) 65%, #ffffff 100%);
    background-size: 220% 100%;
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    animation: sheen 9s ease-in-out infinite alternate;
}
@keyframes sheen { to { background-position: 100% 0; } }
.hero-sub { font-size: 0.95rem; color: var(--text-muted); margin: 0.6rem 0 1.1rem; }
 
/* ── Waveform (the signature element) ── */
.wave { display: flex; align-items: center; gap: 3px; height: 38px; }
.wave.small { height: 22px; gap: 2px; }
.wave i {
    display: block; width: 3px; height: 100%; border-radius: 3px;
    background: linear-gradient(var(--accent-glow), var(--accent-2));
    transform-origin: center; transform: scaleY(0.2);
    animation: eq var(--dur) ease-in-out infinite alternate; animation-delay: var(--d);
}
@keyframes eq { from { transform: scaleY(0.15); } to { transform: scaleY(var(--h)); } }
 
/* ── Cards ── */
.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.35rem 1.5rem; margin-bottom: 1rem; position: relative; overflow: hidden;
    transition: border-color .2s;
    --c: var(--accent);
}
.card:hover { border-color: var(--c); }
.card::before { content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: var(--c); }
.card.c-cyan   { --c: var(--accent-2); }
.card.c-purple { --c: var(--accent-glow); }
.card.c-amber  { --c: var(--warning); }
.card-title { font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700; color: var(--text-muted); margin-bottom: 0.7rem; }
.card-content { font-size: 0.92rem; line-height: 1.75; color: var(--text); }
.card-content p { margin: 0 0 0.6rem; }
.card-content ul, .card-content ol { padding-left: 1.2rem; margin: 0 0 0.5rem; }
.card-content li { margin-bottom: 0.35rem; }
.card-content li::marker { color: var(--c); }
.card-content .md-h { font-family: 'Syne', sans-serif; font-weight: 700; margin-top: 0.4rem; }
.card-content strong { color: #fff; }
.card-content code { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
.muted { color: var(--text-muted); }
.session-title { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--text); }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700; margin: 0 0 1rem; }
 
/* ── Stats strip ── */
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.9rem 1.1rem; }
.stat-num { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.9rem; line-height: 1.1; counter-reset: num var(--target); }
.stat-num::after { content: counter(num); }
.stat-num.anim { counter-reset: num var(--n); animation: count 1.3s cubic-bezier(.2,.7,.2,1) both; }
@keyframes count { from { --n: 0; } to { --n: var(--target); } }
.stat-unit { font-size: 0.8rem; color: var(--text-muted); margin-left: 4px; font-weight: 500; }
.stat-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
 
/* ── One orchestrated reveal when results first appear ── */
.reveal.anim { opacity: 0; animation: rise .65s cubic-bezier(.2,.7,.2,1) forwards; animation-delay: calc(var(--i, 0) * 110ms); }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
 
/* ── Live pipeline tracker ── */
.tracker { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1.2rem; }
.thead { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.9rem; gap: 1rem; }
.thead-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.05rem; }
.ptrack { height: 5px; background: var(--surface-2); border-radius: 99px; overflow: hidden; margin-bottom: 1rem; }
.pfill {
    height: 100%; width: var(--to); border-radius: 99px; position: relative; overflow: hidden;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    animation: fillbar .7s cubic-bezier(.3,.7,.2,1) both;
}
@keyframes fillbar { from { width: var(--from); } to { width: var(--to); } }
.pfill::after { content: ''; position: absolute; inset: 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,.45), transparent); transform: translateX(-100%); animation: shimmer 1.4s infinite; }
@keyframes shimmer { to { transform: translateX(100%); } }
.trow { display: flex; align-items: center; gap: 0.9rem; padding: 0.55rem 0; opacity: .45; transition: opacity .3s; }
.trow.active, .trow.done, .trow.error { opacity: 1; }
.tdot { width: 22px; height: 22px; border-radius: 50%; border: 2px solid var(--border); flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 12px; color: #fff; }
.tdot.active { border-color: var(--accent-glow); background: rgba(124,58,237,.25); animation: ring 1.4s infinite; }
.tdot.done { background: var(--success); border-color: var(--success); }
.tdot.done::after { content: '✓'; font-weight: 700; animation: pop .35s ease-out; }
.tdot.error { background: var(--danger); border-color: var(--danger); }
.tdot.error::after { content: '!'; font-weight: 700; }
@keyframes ring { 0% { box-shadow: 0 0 0 0 rgba(159,103,255,.55); } 100% { box-shadow: 0 0 0 10px rgba(159,103,255,0); } }
@keyframes pop { from { transform: scale(0); } 60% { transform: scale(1.4); } to { transform: scale(1); } }
.tinfo { flex: 1; }
.tlabel { font-size: 0.92rem; font-weight: 500; }
.tdesc { font-size: 0.76rem; color: var(--text-muted); }
.tright { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: var(--text-muted); }
.trow.active .tright { color: var(--accent-glow); animation: blink 1.2s infinite; }
@keyframes blink { 50% { opacity: .35; } }
.terr { margin-top: .8rem; padding: .7rem .9rem; border-radius: 8px; background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.35); color: #fca5a5; font-size: .82rem; word-break: break-word; }
 
/* ── Sidebar bits ── */
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.72rem; font-weight: 600; margin-bottom: .4rem; }
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
.srow { display: grid; grid-template-columns: 1fr auto; gap: 2px 8px; font-size: 0.78rem; margin-bottom: .55rem; }
.srow b { font-family: 'JetBrains Mono', monospace; font-weight: 500; color: var(--text-muted); }
.mini { grid-column: 1 / -1; height: 3px; background: var(--surface-2); border-radius: 3px; overflow: hidden; }
.mini i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); animation: grow .9s cubic-bezier(.3,.7,.2,1) both; }
@keyframes grow { from { width: 0; } }
 
/* ── Inputs & buttons ── */
.stTextInput input, [data-baseweb="select"] > div {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
}
.stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important; }
[data-testid="stFileUploader"] section { background: var(--surface-2) !important; border: 1px dashed var(--border) !important; border-radius: 10px !important; }
 
[data-testid^="stBaseButton-primary"], button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
    position: relative; overflow: hidden; transition: transform .2s, box-shadow .2s !important;
}
[data-testid^="stBaseButton-primary"]:hover, button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-1px); box-shadow: 0 8px 25px rgba(124,58,237,0.45) !important;
}
[data-testid^="stBaseButton-primary"]::after, button[kind="primary"]::after, button[kind="primaryFormSubmit"]::after {
    content: ''; position: absolute; inset: 0; background: linear-gradient(105deg, transparent 35%, rgba(255,255,255,.25) 50%, transparent 65%);
    transform: translateX(-120%); transition: transform .6s;
}
[data-testid^="stBaseButton-primary"]:hover::after, button[kind="primary"]:hover::after, button[kind="primaryFormSubmit"]:hover::after { transform: translateX(120%); }
[data-testid^="stBaseButton-secondary"], button[kind="secondary"] {
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important; transition: border-color .2s, color .2s !important;
}
[data-testid^="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover { border-color: var(--accent) !important; color: var(--accent-glow) !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
 
/* ── Expander ── */
[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 12px; }
[data-testid="stExpander"] summary { font-family: 'Syne', sans-serif; font-weight: 700; }
 
/* ── Transcript ── */
.transcript-box {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 1.1rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.85;
    max-height: 340px; overflow-y: auto; color: var(--text-muted); word-break: break-word;
}
.transcript-box mark { background: rgba(124,58,237,.45); color: #fff; border-radius: 3px; padding: 0 2px; }
 
/* ── Chat ── */
.chat-container {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.1rem; max-height: 460px; overflow-y: auto; margin-bottom: 1rem;
    display: flex; flex-direction: column-reverse;   /* keeps newest message in view */
}
.chat-inner { display: flex; flex-direction: column; gap: 1rem; }
.chat-msg { display: flex; flex-direction: column; gap: 0.25rem; }
.chat-msg.user { align-items: flex-end; }
.chat-msg.bot  { align-items: flex-start; }
.chat-label { font-size: 0.72rem; font-weight: 600; }
.user .chat-label { color: var(--accent-glow); }
.bot  .chat-label { color: var(--accent-2); }
.chat-bubble { padding: 0.65rem 1rem; border-radius: 12px; font-size: 0.9rem; line-height: 1.65; max-width: 88%; animation: bubble .3s ease-out; }
.chat-bubble p { margin: 0 0 .4rem; } .chat-bubble p:last-child { margin: 0; }
.chat-bubble ul, .chat-bubble ol { margin: 0 0 .4rem; padding-left: 1.1rem; }
.user .chat-bubble { background: rgba(124,58,237,0.16); border: 1px solid rgba(124,58,237,0.3); border-bottom-right-radius: 3px; }
.bot  .chat-bubble { background: rgba(6,182,212,0.09); border: 1px solid rgba(6,182,212,0.22); border-bottom-left-radius: 3px; }
@keyframes bubble { from { opacity: 0; transform: translateY(6px) scale(.98); } to { opacity: 1; transform: none; } }
.dots { display: inline-flex; gap: 5px; padding: 4px 0; }
.dots i { width: 7px; height: 7px; border-radius: 50%; background: var(--accent-2); animation: bounce 1.1s infinite; }
.dots i:nth-child(2) { animation-delay: .15s; } .dots i:nth-child(3) { animation-delay: .3s; }
@keyframes bounce { 0%, 80%, 100% { transform: translateY(0); opacity: .4; } 40% { transform: translateY(-6px); opacity: 1; } }
.cursor { display: inline-block; width: 7px; height: 1em; margin-left: 2px; vertical-align: text-bottom; background: var(--accent-2); animation: blink 0.8s steps(1) infinite; }
.chat-empty { text-align: center; padding: 1.6rem 1rem; }
.chat-empty .big { font-size: 1.8rem; margin-bottom: .3rem; }
 
/* ── Empty state ── */
.empty { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 3rem 1rem 1rem; }
.empty h2 { font-family: 'Syne', sans-serif; font-size: 1.7rem; margin: 1.4rem 0 .5rem; }
.empty p { color: var(--text-muted); max-width: 460px; line-height: 1.7; font-size: .92rem; }
.feats { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 2.2rem; width: 100%; text-align: left; }
.feat { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; transition: border-color .2s, transform .2s; }
.feat:hover { border-color: var(--accent); transform: translateY(-3px); }
.feat .ic { font-size: 1.4rem; } .feat b { display: block; font-family: 'Syne', sans-serif; margin: .5rem 0 .25rem; }
.feat span { color: var(--text-muted); font-size: .84rem; line-height: 1.6; }
 
.footer { text-align: center; color: var(--text-muted); font-size: .78rem; margin-top: 3rem; padding-bottom: 1rem; }
.footer a { color: var(--accent-glow); text-decoration: none; }
 
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
 
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
</style>
""", unsafe_allow_html=True)
 
# ─── Warm up the embedding model in the background (once per server process) ───
@st.cache_resource(show_spinner=False)
def _warm_up():
    warm_up_embeddings()
    return True
 
 
_warm_up()
 
# ─── Constants ──────────────────────────────────────────────────────────────────
STEPS = [
    ("audio",      "🔊", "Audio processing", "Fetching and chunking the audio"),
    ("transcript", "📝", "Transcription",    "Converting speech to text"),
    ("title",      "🏷️", "Title generation", "Naming the session"),
    ("summary",    "📋", "Summarisation",    "Condensing the transcript"),
    ("extract",    "🔍", "Extraction",       "Action items, decisions, open questions"),
    ("rag",        "🧠", "RAG index",        "Building the searchable index"),
]
SUGGESTIONS = [
    "Summarise the key points",
    "What needs follow-up?",
    "What was left unresolved?",
]
 
# ─── Session State ──────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "timings": {},
    "revealed": False,
    "pending_q": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
 
 
# ─── Helpers ────────────────────────────────────────────────────────────────────
def H(s: str) -> str:
    """Collapse a multi-line HTML template so Streamlit's markdown parser never
    mistakes indented lines or blank lines for code blocks."""
    return "".join(line.strip() for line in s.strip().splitlines())
 
 
def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))
 
 
def _inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s
 
 
def md_to_html(text) -> str:
    """Tiny, safe markdown renderer (bullets, numbers, bold, code) for LLM output."""
    text = html.escape(str(text or "").strip())
    out, list_tag = [], None
    for line in text.splitlines():
        s = line.strip()
        mb = re.match(r"^[-*•]\s+(.*)", s)
        mn = re.match(r"^\d+[.)]\s+(.*)", s)
        if mb or mn:
            tag = "ul" if mb else "ol"
            if list_tag != tag:
                if list_tag:
                    out.append(f"</{list_tag}>")
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{_inline((mb or mn).group(1))}</li>")
            continue
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None
        if not s:
            continue
        hm = re.match(r"^#{1,6}\s+(.*)", s)
        out.append(f"<p class='md-h'>{_inline(hm.group(1))}</p>" if hm else f"<p>{_inline(s)}</p>")
    if list_tag:
        out.append(f"</{list_tag}>")
    return "".join(out) or "<p class='muted'>Nothing found.</p>"
 
 
def count_items(text) -> int:
    lines = [l.strip() for l in str(text or "").splitlines() if l.strip()]
    bullets = [l for l in lines if re.match(r"^([-*•]|\d+[.)])\s+", l)]
    return len(bullets) or len(lines)
 
 
def fmt_t(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 60 else f"{int(sec // 60)}m {int(sec % 60)}s"
 
 
def with_retry(fn, *args, retries: int = 5):
    """Retry a pipeline step when Groq returns a 429 rate-limit error."""
    for attempt in range(retries):
        try:
            return fn(*args)
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "rate_limit" not in msg:
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", msg)
            wait = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1) if m else 2 ** attempt * 2
            time.sleep(wait)
    raise RuntimeError("Rate limit: retries exhausted")
 
 
def wave_html(n: int = 34, fast: bool = False, small: bool = False) -> str:
    rnd = random.Random(7)
    bars = "".join(
        f'<i style="--d:-{rnd.uniform(0, 2):.2f}s;'
        f'--dur:{rnd.uniform(0.7, 1.5) * (0.55 if fast else 1):.2f}s;'
        f'--h:{rnd.uniform(0.35, 1):.2f}"></i>'
        for _ in range(n)
    )
    return f'<div class="wave{" small" if small else ""}">{bars}</div>'
 
 
def render_tracker(states: dict, timings: dict, pct_from: float, pct_to: float, err: str = None) -> str:
    rows = []
    for key, icon, label, desc in STEPS:
        s = states.get(key, "pending")
        right = {"done": fmt_t(timings.get(key, 0)), "active": "running…", "error": "failed", "pending": ""}[s]
        rows.append(
            f'<div class="trow {s}"><div class="tdot {s}"></div>'
            f'<div class="tinfo"><div class="tlabel">{icon} {label}</div><div class="tdesc">{desc}</div></div>'
            f'<div class="tright">{right}</div></div>'
        )
    if err:
        head, wave = "Analysis stopped", ""
    elif pct_to >= 100:
        head, wave = "Analysis complete", ""
    else:
        head, wave = "Analysing your video", wave_html(16, fast=True, small=True)
    err_html = f'<div class="terr">{esc(err)}</div>' if err else ""
    return H(f"""
    <div class="tracker">
      <div class="thead"><div class="thead-title">{head}</div>{wave}</div>
      <div class="ptrack"><div class="pfill" style="--from:{pct_from:.0f}%;--to:{pct_to:.0f}%"></div></div>
      {"".join(rows)}
      {err_html}
    </div>""")
 
 
def chat_html(history, pending_user=None, partial=None, thinking=False, cursor=False) -> str:
    def user(t):
        return (f'<div class="chat-msg user"><span class="chat-label">You</span>'
                f'<div class="chat-bubble">{esc(t)}</div></div>')
 
    def bot(inner):
        return (f'<div class="chat-msg bot"><span class="chat-label">🤖 Assistant</span>'
                f'<div class="chat-bubble">{inner}</div></div>')
 
    parts = []
    for m in history:
        parts.append(user(m["content"]) if m["role"] == "user" else bot(md_to_html(m["content"])))
    if pending_user:
        parts.append(user(pending_user))
        if thinking:
            parts.append(bot('<span class="dots"><i></i><i></i><i></i></span>'))
        elif partial is not None:
            body = md_to_html(partial)
            if cursor:
                tag = '<span class="cursor"></span>'
                body = body[:-4] + tag + body[-4:] if body.endswith("</p>") else body + tag
            parts.append(bot(body))
    return H(f'<div class="chat-container"><div class="chat-inner">{"".join(parts)}</div></div>')
 
 
def typewriter(ph, history, question, answer):
    words = answer.split(" ")
    step = max(1, len(words) // 90)          # long answers type in bigger chunks
    delay = min(0.04, 2.5 / max(len(words), 1) * step)
    for i in range(step, len(words) + step, step):
        ph.markdown(chat_html(history, question, " ".join(words[:i]), cursor=True), unsafe_allow_html=True)
        time.sleep(delay)
 
 
def queue_question(q: str):
    st.session_state.pending_q = q
 
 
def save_upload(f) -> str:
    suffix = Path(f.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getbuffer())
    tmp.close()
    return tmp.name
 
 
def build_report(r: dict) -> str:
    return (
        f"# {r['title']}\n\n## Summary\n{r['summary']}\n\n"
        f"## Action Items\n{r['action_items']}\n\n"
        f"## Key Decisions\n{r['key_decisions']}\n\n"
        f"## Open Questions\n{r['open_questions']}\n"
    )
 
 
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:40] or "report"
 
 
# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(H('<div class="hero-title" style="font-size:1.7rem">🎬 AI<br>Video</div>'), unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:.6rem">Meeting intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
 
    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    mode = st.radio("Source", ["YouTube URL / path", "Upload file"], horizontal=True, label_visibility="collapsed")
    source, uploaded = "", None
    if mode == "Upload file":
        uploaded = st.file_uploader(
            "Video or audio file", type=["mp4", "mkv", "mov", "webm", "mp3", "wav", "m4a"], label_visibility="collapsed"
        )
    else:
        source = st.text_input("YouTube URL or file path", placeholder="https://youtube.com/watch?v=…")
 
    language = st.selectbox("Language", ["english", "hinglish"], index=0, format_func=str.title)
    run_btn = st.button("⚡  Analyse", use_container_width=True, type="primary")
 
    if st.session_state.pipeline_done and st.session_state.timings:
        timings = st.session_state.timings
        total = sum(timings.values()) or 1
        st.markdown("---")
        st.markdown(f'<span class="badge badge-green">Pipeline · {fmt_t(total)}</span>', unsafe_allow_html=True)
        rows = "".join(
            f'<div class="srow"><span>{icon} {label}</span><b>{fmt_t(timings.get(k, 0))}</b>'
            f'<div class="mini"><i style="width:{timings.get(k, 0) / total * 100:.0f}%"></i></div></div>'
            for k, icon, label, _ in STEPS
        )
        st.markdown(H(rows), unsafe_allow_html=True)
 
        r_ = st.session_state.result
        if r_:
            st.markdown("---")
            st.download_button(
                "⬇  Download report (.md)", build_report(r_), file_name=f"{slug(r_['title'])}.md",
                mime="text/markdown", use_container_width=True,
            )
            st.download_button(
                "⬇  Download transcript (.txt)", r_["transcript"], file_name=f"{slug(r_['title'])}-transcript.txt",
                mime="text/plain", use_container_width=True,
            )
 
# ─── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe, summarise, and ask questions about any meeting or video.</div>', unsafe_allow_html=True)
st.markdown(wave_html(44), unsafe_allow_html=True)
st.markdown("---")
 
# ─── Run Pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    if mode == "Upload file" and uploaded is None:
        st.error("Upload a video or audio file first.")
    elif mode != "Upload file" and not source.strip():
        st.error("Enter a YouTube URL or a file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pending_q = None
        st.session_state.timings = {}
        st.session_state.revealed = False
 
        states = {k: "pending" for k, *_ in STEPS}
        timings = {}
        prog = {"pct": 0.0}
        tracker_ph = st.empty()
 
        def draw(err=None):
            done = sum(v == "done" for v in states.values())
            active = any(v == "active" for v in states.values())
            pct = (done + (0.5 if active else 0)) / len(STEPS) * 100
            tracker_ph.markdown(render_tracker(states, timings, prog["pct"], pct, err), unsafe_allow_html=True)
            prog["pct"] = pct
 
        def run_step(key, fn, *args):
            states[key] = "active"
            draw()
            t0 = time.perf_counter()
            out = with_retry(fn, *args)
            timings[key] = time.perf_counter() - t0
            states[key] = "done"
            draw()
            return out
 
        tmp_path, success = None, False
        try:
            draw()
            src = source.strip()
            if uploaded is not None:
                tmp_path = src = save_upload(uploaded)
 
            t_start = time.perf_counter()
            chunks       = run_step("audio", process_input, src)
            transcript   = run_step("transcript", transcribe_all, chunks, language)
            title        = run_step("title", generate_title, transcript)
            summary      = run_step("summary", summarize, transcript)
 
            action_items, decisions, questions = run_step("extract", extract_all, transcript)
            rag_chain    = run_step("rag", build_rag_chain, transcript)
 
            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "elapsed": time.perf_counter() - t_start,
            }
            st.session_state.timings = timings
            st.session_state.pipeline_done = True
            success = True
        except Exception as e:
            for k, v in states.items():
                if v == "active":
                    states[k] = "error"
            draw(err=str(e))
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
 
        if success:
            time.sleep(0.7)      # let the "complete" state register
            st.rerun()
 
# ─── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    first = not st.session_state.revealed          # play the entrance only once
    A = " anim" if first else ""
 
    def rv(i: int) -> str:
        return f'style="--i:{i}"'
 
    # Title
    st.markdown(H(f"""
    <div class="card reveal{A}" {rv(0)}>
      <div class="card-title">📌 Session title</div>
      <div class="session-title">{esc(r['title'])}</div>
    </div>"""), unsafe_allow_html=True)
 
    # Stats
    words = len(r["transcript"].split())
    stats = [
        (words, "", "Words transcribed"),
        (max(1, round(words / 150)), "min", "Estimated talk time"),
        (count_items(r["action_items"]), "", "Action items"),
        (count_items(r["key_decisions"]), "", "Key decisions"),
        (max(1, round(r.get("elapsed", 0))), "s", "Processing time"),
    ]
    cells = "".join(
        f'<div class="stat reveal{A}" {rv(1 + i)}>'
        f'<div><span class="stat-num{A}" style="--target:{n}"></span><span class="stat-unit">{u}</span></div>'
        f'<div class="stat-label">{lbl}</div></div>'
        for i, (n, u, lbl) in enumerate(stats)
    )
    st.markdown(H(f'<div class="stats">{cells}</div>'), unsafe_allow_html=True)
 
    # Summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        st.markdown(H(f"""
        <div class="card reveal{A}" {rv(5)}>
          <div class="card-title">📋 Summary</div>
          <div class="card-content">{md_to_html(r['summary'])}</div>
        </div>"""), unsafe_allow_html=True)
 
    with col2:
        query = st.session_state.get("tsearch", "").strip()
        with st.expander("📝 Full transcript", expanded=bool(query)):
            st.text_input("Search transcript", key="tsearch", placeholder="Search for a word or phrase…")
            body = esc(r["transcript"])
            hits = 0
            if query:
                body, hits = re.subn(f"({re.escape(esc(query))})", r"<mark>\1</mark>", body, flags=re.I)
            body = body.replace("\n", "<br>")
            if query:
                st.caption(f"{hits} match{'es' if hits != 1 else ''}")
            st.markdown(f'<div class="transcript-box">{body}</div>', unsafe_allow_html=True)
 
    # Insights
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, cls, icon, title, key, i in [
        (c1, "c-cyan",   "✅", "Action items",   "action_items",   6),
        (c2, "c-purple", "🔑", "Key decisions",  "key_decisions",  7),
        (c3, "c-amber",  "❓", "Open questions", "open_questions", 8),
    ]:
        with col:
            st.markdown(H(f"""
            <div class="card {cls} reveal{A}" {rv(i)}>
              <div class="card-title">{icon} {title}</div>
              <div class="card-content">{md_to_html(r[key])}</div>
            </div>"""), unsafe_allow_html=True)
 
    st.markdown("---")
 
    # ── RAG Chat ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💬 Chat with your meeting</div>', unsafe_allow_html=True)
 
    history = st.session_state.chat_history
    pending = st.session_state.pending_q
    chat_ph = st.empty()
 
    if pending:
        chat_ph.markdown(chat_html(history, pending, thinking=True), unsafe_allow_html=True)
        try:
            answer = str(ask_question(r["rag_chain"], pending))
        except Exception as e:
            answer = f"Sorry, I couldn't answer that: {e}"
        typewriter(chat_ph, history, pending, answer)
        history.append({"role": "user", "content": pending})
        history.append({"role": "assistant", "content": answer})
        st.session_state.pending_q = None
        st.session_state.revealed = True
        st.rerun()
    elif history:
        chat_ph.markdown(chat_html(history), unsafe_allow_html=True)
    else:
        chat_ph.markdown(H("""
        <div class="card chat-empty">
          <div class="big">💬</div>
          <div class="muted" style="font-size:.88rem">Ask anything about the transcript. Answers come from what was actually said.</div>
        </div>"""), unsafe_allow_html=True)
        sc = st.columns(len(SUGGESTIONS), gap="small")
        for i, s in enumerate(SUGGESTIONS):
            sc[i].button(s, key=f"chip{i}", on_click=queue_question, args=(s,), use_container_width=True)
 
    with st.form("chat_form", clear_on_submit=True):
        f1, f2 = st.columns([6, 1], gap="small")
        q = f1.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
        sent = f2.form_submit_button("Send", use_container_width=True, type="primary")
    if sent and q.strip():
        queue_question(q.strip())
        st.rerun()
 
    if history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()
 
    st.session_state.revealed = True
 
elif not run_btn:
    # ── Empty state ─────────────────────────────────────────────────────────────
    st.markdown(H("""
    <div class="empty">
      <h2>Ready when you are</h2>
      <p>Paste a YouTube link, a file path, or upload a recording in the sidebar. Pick a language and press Analyse.</p>
      <div class="feats">
        <div class="feat"><div class="ic">📝</div><b>Transcribe</b><span>Speech to text in English or Hinglish, even for long recordings.</span></div>
        <div class="feat"><div class="ic">📋</div><b>Summarise</b><span>A title, summary, action items, key decisions and open questions.</span></div>
        <div class="feat"><div class="ic">💬</div><b>Ask</b><span>Chat with the transcript and get answers grounded in what was said.</span></div>
      </div>
    </div>"""), unsafe_allow_html=True)
 
st.markdown(
    f'<div class="footer">Built by {esc(AUTHOR)} · <a href="{esc(GITHUB_URL)}" target="_blank">GitHub</a></div>',
    unsafe_allow_html=True,
)