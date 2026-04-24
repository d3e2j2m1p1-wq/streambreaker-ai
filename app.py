"""StreamBreaker AI — v12 · Editorial Design"""
import streamlit as st
import sys, os, numpy as np, math, re, json, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import StreamBreakerPipeline
from model0_audio import extract_features as extract_audio_features, get_file_metadata, KEY_NAMES

st.set_page_config(page_title="StreamBreaker AI", page_icon="🎵",
                   layout="wide", initial_sidebar_state="collapsed")

for k, v in [
    ("result",None),("af",None),("auto_detected",False),
    ("auto_lyrics",""),("lyrics_source",""),("dark_mode",False),("page","analyze"),
    ("profile_saved",False),("instagram_val",1200),("spotify_val",350),
    ("youtube_val",800),("career_val","emerging"),("budget_val",1500),
    ("openai_key",""),("claude_key",""),("groq_key",""),
    ("chat_history",[]),("_run_af",None),("_run_genre","indie-pop"),
    ("_run_backend","openai"),("active_week",None),("report_budget",1500),
]:
    if k not in st.session_state:
        st.session_state[k] = v

page = st.session_state.page
dark = st.session_state.dark_mode

if dark:
    T = {
        "bg":"#0d0b09","surface":"#161310","surface2":"#1e1a16","surface3":"#252018",
        "border":"#2e2820","border2":"#3a3028","border3":"#4a3e30",
        "ink":"#e8e0d4","ink2":"#9a8e80","ink3":"#5a5048","ink4":"#3a3028",
        "accent":"#c8392b","accent2":"#e05040","gold":"#c8902b","gold2":"#e0b040",
        "blue":"#1a3a5c","blue2":"#2a5a8c",
        "input_bg":"#1e1a16","input_border":"#3a3028","input_color":"#e8e0d4",
        "pill_bg":"#2e2820","tog":"☀️  Light Mode",
        "noise":"rgba(255,255,255,.015)",
    }
else:
    T = {
        "bg":"#f5f0e8","surface":"#faf7f2","surface2":"#ede8dc","surface3":"#e0d8cc",
        "border":"#d8d0be","border2":"#c8bfaa","border3":"#b8ae98",
        "ink":"#1a1510","ink2":"#5a5040","ink3":"#9a9080","ink4":"#c0b8a8",
        "accent":"#c8392b","accent2":"#a82010","gold":"#a87020","gold2":"#c89030",
        "blue":"#1a3a5c","blue2":"#2a5a8c",
        "input_bg":"#faf7f2","input_border":"#d0c8b8","input_color":"#1a1510",
        "pill_bg":"#ede8dc","tog":"🌙  Dark Mode",
        "noise":"rgba(0,0,0,.02)",
    }

# ── CHART HELPERS ──────────────────────────────────────────

def gauge_editorial(pct):
    c = T["gold"] if pct >= 70 else T["accent"] if pct >= 50 else "#888"
    label = "HIGH POTENTIAL" if pct >= 70 else "MODERATE POTENTIAL" if pct >= 50 else "LOW POTENTIAL"
    r=58; cx=80; cy=72; sw=8
    arc = math.pi * r
    fill = arc * (pct / 100)
    gap  = arc - fill
    return f"""<svg viewBox="0 0 160 90" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:180px;display:block;">
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
    fill="none" stroke="{T["border2"]}" stroke-width="{sw}" stroke-linecap="round"/>
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
    fill="none" stroke="{c}" stroke-width="{sw}" stroke-linecap="round"
    stroke-dasharray="{fill:.1f} {gap:.1f}"/>
  <line x1="{cx}" y1="{cy}" x2="{cx - r*0.6*math.cos(math.pi*(1-pct/100)):.1f}"
    y2="{cy - r*0.6*math.sin(math.pi*(1-pct/100)):.1f}"
    stroke="{c}" stroke-width="2" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="4" fill="{T["surface"]}" stroke="{c}" stroke-width="2"/>
  <text x="{cx}" y="{cy-8}" text-anchor="middle" font-family="Playfair Display,serif"
    font-size="22" font-weight="700" fill="{c}">{pct:.0f}%</text>
  <text x="{cx}" y="{cy+14}" text-anchor="middle" font-family="IBM Plex Mono,monospace"
    font-size="7" fill="{T["ink3"]}" letter-spacing="1">{label}</text>
</svg>"""

def feature_bars_svg(af):
    feats = [
        ("Danceability", af["danceability"], 0.65),
        ("Energy",       af["energy"],       0.70),
        ("Valence",      af["valence"],       0.50),
        ("Acousticness", af["acousticness"],  0.25),
        ("Liveness",     af["liveness"],      0.15),
        ("Speechiness",  af["speechiness"],   0.05),
    ]
    W=280; row=22; pad_l=84; pad_r=32; bar_h=2
    H = len(feats)*row + 8
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">']
    for i,(name,val,bench) in enumerate(feats):
        y = 8 + i*row
        bar_w = W - pad_l - pad_r
        fill_w = int(bar_w * min(1.0, max(0.0, val)))
        bench_x = pad_l + int(bar_w * bench)
        clr = T["gold"] if val >= bench else T["accent"]
        # Label
        parts.append(f'<text x="{pad_l-6}" y="{y+bar_h//2+4}" text-anchor="end" font-family="IBM Plex Mono,monospace" font-size="9" fill="{T["ink3"]}">{name}</text>')
        # Track
        parts.append(f'<rect x="{pad_l}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{T["border2"]}"/>')
        # Fill
        parts.append(f'<rect x="{pad_l}" y="{y}" width="{fill_w}" height="{bar_h}" fill="{clr}"/>')
        # Benchmark tick
        parts.append(f'<rect x="{bench_x-1}" y="{y-3}" width="1" height="{bar_h+6}" fill="{T["ink4"]}"/>')
        # Value
        parts.append(f'<text x="{W-pad_r+4}" y="{y+bar_h//2+4}" font-family="IBM Plex Mono,monospace" font-size="9" font-weight="500" fill="{clr}">{val:.2f}</text>')
    parts.append('</svg>')
    return "".join(parts)

def projection_svg_ed(pp, budget):
    cps = 0.022
    total_lo = int(budget/cps*(pp/100)*0.7)
    total_hi = int(budget/cps*(pp/100)*1.3)
    splits   = [0.15, 0.35, 0.65, 1.0]
    lo_pts   = [int(total_lo*s) for s in splits]
    hi_pts   = [int(total_hi*s) for s in splits]
    maxv     = hi_pts[-1] or 1
    W=400; H=120; pl=40; pb=20; pt=8; pr=8
    cw=W-pl-pr; ch=H-pb-pt
    xs=[pl+int(cw*i/3) for i in range(4)]
    weeks=["Week 1","Week 2","Week 3","Week 4"]
    parts=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">']
    for frac in [0.25,0.5,0.75,1.0]:
        y=H-pb-int(ch*frac)
        parts.append(f'<line x1="{pl}" y1="{y}" x2="{W-pr}" y2="{y}" stroke="{T["border"]}" stroke-width="1"/>')
        parts.append(f'<text x="{pl-4}" y="{y+3}" text-anchor="end" font-family="IBM Plex Mono,monospace" font-size="7" fill="{T["ink4"]}">{int(maxv*frac/1000)}k</text>')
    # area
    hi_c=" ".join([f"{xs[i]},{H-pb-int(ch*hi_pts[i]/maxv)}" for i in range(4)])
    lo_c=" ".join([f"{xs[i]},{H-pb-int(ch*lo_pts[i]/maxv)}" for i in range(3,-1,-1)])
    fill_col = "rgba(200,57,43,.08)" if not dark else "rgba(200,57,43,.12)"
    parts.append(f'<polygon points="{hi_c} {lo_c}" fill="{fill_col}"/>')
    # hi line
    parts.append(f'<polyline points="{hi_c}" fill="none" stroke="{T["accent"]}" stroke-width="1.5" stroke-linejoin="round"/>')
    # lo dashed
    lo_l=" ".join([f"{xs[i]},{H-pb-int(ch*lo_pts[i]/maxv)}" for i in range(4)])
    parts.append(f'<polyline points="{lo_l}" fill="none" stroke="{T["ink3"]}" stroke-width="1" stroke-dasharray="4,3" stroke-linejoin="round"/>')
    for i in range(4):
        x=xs[i]; y_hi=H-pb-int(ch*hi_pts[i]/maxv)
        parts.append(f'<circle cx="{x}" cy="{y_hi}" r="3" fill="{T["accent"]}"/>')
        parts.append(f'<text x="{x}" y="{H-pb+13}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="7" fill="{T["ink3"]}">{weeks[i]}</text>')
        if i==3:
            parts.append(f'<text x="{x+4}" y="{y_hi-6}" font-family="IBM Plex Mono,monospace" font-size="8" fill="{T["accent"]}">{total_hi//1000}k</text>')
    parts.append('</svg>')
    return "".join(parts)

def budget_donut_svg(budget):
    slices=[("Spotify","#6c63ff",0.65),("TikTok",T["accent"],0.25),("Reserve",T["gold"],0.10)]
    cx=65;cy=65;r=45;sw=18
    circ=2*math.pi*r
    bud_str=f"{budget:,}"
    parts=[f'<svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:220px;display:block;">']
    offset=0
    for lbl,col,pct in slices:
        d=circ*pct; g=circ-d
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-dasharray="{d:.1f} {g:.1f}" stroke-dashoffset="{-offset:.1f}" transform="rotate(-90 {cx} {cy})" opacity=".85"/>')
        offset+=circ*pct
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r-sw//2-2}" fill="{T["surface"]}"/>')
    parts.append(f'<text x="{cx}" y="{cy-3}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="11" font-weight="500" fill="{T["ink"]}">${bud_str}</text>')
    parts.append(f'<text x="{cx}" y="{cy+11}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="7" fill="{T["ink3"]}" letter-spacing="1">TOTAL</text>')
    lx=138
    for i,(lbl,col,pct) in enumerate(slices):
        ly=28+i*30
        amt=f"{int(budget*pct):,}"
        parts.append(f'<rect x="{lx}" y="{ly}" width="6" height="6" fill="{col}"/>')
        parts.append(f'<text x="{lx+10}" y="{ly+6}" font-family="IBM Plex Serif,serif" font-size="10" fill="{T["ink2"]}">{lbl}</text>')
        parts.append(f'<text x="{lx+10}" y="{ly+17}" font-family="IBM Plex Mono,monospace" font-size="9" fill="{T["ink3"]}">${amt}</text>')
    parts.append('</svg>')
    return "".join(parts)

def fetch_lyrics_gpt(title, artist, api_key):
    if not api_key or not title: return "", "not_found"
    try:
        query = f"{artist} - {title}" if artist else title
        prompt = (f"Provide the full lyrics for: {query}\n\n"
                  "Format with section headers like [Verse 1], [Chorus], [Bridge].\n"
                  "If unknown, reply only: UNKNOWN\nReturn only lyrics or UNKNOWN, no other text.")
        payload = json.dumps({"model":"gpt-3.5-turbo","max_tokens":1500,
            "messages":[{"role":"user","content":prompt}]}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
            data=payload, headers={"Content-Type":"application/json","Authorization":f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"].strip()
        if "UNKNOWN" in text[:30] or len(text)<30: return "", "not_found"
        return text, "gpt"
    except: return "", "not_found"

def chat_call(messages, system_prompt):
    claude_key = st.session_state.claude_key
    openai_key = st.session_state.openai_key or os.getenv("OPENAI_API_KEY","")
    try:
        if not openai_key:
            try: openai_key = st.secrets.get("OPENAI_API_KEY","")
            except: pass
    except: pass
    groq_key = st.session_state.groq_key
    if claude_key:
        payload = json.dumps({"model":"claude-sonnet-4-20250514","max_tokens":800,
            "system":system_prompt,"messages":messages}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type":"application/json","x-api-key":claude_key,"anthropic-version":"2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["content"][0]["text"].strip()
    elif openai_key:
        full = [{"role":"system","content":system_prompt}]+messages
        payload = json.dumps({"model":"gpt-3.5-turbo","max_tokens":800,"messages":full}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload,
            headers={"Content-Type":"application/json","Authorization":f"Bearer {openai_key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    elif groq_key:
        full = [{"role":"system","content":system_prompt}]+messages
        payload = json.dumps({"model":"llama3-8b-8192","max_tokens":800,"messages":full}).encode()
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=payload,
            headers={"Content-Type":"application/json","Authorization":f"Bearer {groq_key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    raise Exception("No API key found — add one in ⚙️ Settings")

@st.cache_resource
def load_pipeline(backend="openai", api_key=None):
    return StreamBreakerPipeline(backend=backend, api_key=api_key)

# ── EDITORIAL CSS ─────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400;1,700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*{{box-sizing:border-box;}}
html,body,[class*="css"]{{font-family:'IBM Plex Sans',sans-serif;background:{T["bg"]};color:{T["ink"]};}}
.stApp{{background:{T["bg"]};}}
[data-testid="stHeader"],[data-testid="stToolbar"],footer{{display:none;}}
.main .block-container{{padding:0 0 60px 0!important;max-width:100%!important;}}
[data-testid="stMainBlockContainer"]{{padding:0!important;max-width:100%!important;}}
::-webkit-scrollbar{{width:4px;}}::-webkit-scrollbar-track{{background:{T["bg"]};}}
::-webkit-scrollbar-thumb{{background:{T["border2"]};border-radius:1px;}}

/* ── TOPBAR ── */
.topbar{{background:{T["surface"]};border-bottom:3px solid {T["accent"]};padding:0 40px;
  display:flex;align-items:center;height:50px;}}
.topbar-logo{{font-family:'Playfair Display',serif;font-size:17px;font-weight:900;
  color:{T["ink"]};letter-spacing:.02em;margin-right:auto;}}
.topbar-logo span{{color:{T["accent"]};font-style:italic;}}
.topbar-meta{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:{T["ink3"]};
  letter-spacing:.12em;text-transform:uppercase;}}
.dateline{{background:{T["ink"]};color:{T["accent"]};font-family:'IBM Plex Mono',monospace;
  font-size:9px;letter-spacing:.14em;text-transform:uppercase;text-align:center;padding:5px;}}

/* ── NAV TABS ── */
.navrow{{border-bottom:1px solid {T["border"]};padding:0 40px;display:flex;align-items:center;
  background:{T["surface"]};gap:0;}}

/* ── PAGE PADDING ── */
.page-pad{{padding:0 40px;}}

/* ── TYPOGRAPHY ── */
.ed-title{{font-family:'Playfair Display',serif;font-size:52px;font-weight:900;
  color:{T["ink"]};line-height:.95;letter-spacing:-.01em;margin-bottom:8px;}}
.ed-title-sm{{font-family:'Playfair Display',serif;font-size:28px;font-weight:700;
  color:{T["ink"]};line-height:1.1;margin-bottom:6px;}}
.ed-kicker{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:{T["accent"]};
  letter-spacing:.16em;text-transform:uppercase;margin-bottom:10px;}}
.ed-deck{{font-size:13px;color:{T["ink2"]};line-height:1.65;font-style:italic;
  border-left:3px solid {T["accent"]};padding-left:12px;}}
.ed-byline{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:{T["ink3"]};
  letter-spacing:.1em;text-transform:uppercase;margin-top:10px;padding-top:8px;
  border-top:1px solid {T["border"]};}}
.col-rule{{width:1px;background:{T["border"]};margin:0 20px;align-self:stretch;}}

/* ── SECTION HEADERS ── */
.sec-head{{font-family:'IBM Plex Mono',monospace;font-size:8px;color:{T["ink3"]};
  letter-spacing:.16em;text-transform:uppercase;padding-bottom:8px;margin-bottom:14px;
  border-bottom:1px solid {T["border"]};display:flex;justify-content:space-between;}}
.sec-head span{{color:{T["accent"]};}}

/* ── CARDS ── */
.ed-card{{background:{T["surface"]};border:1px solid {T["border"]};padding:16px;margin-bottom:12px;}}
.ed-card-sm{{background:{T["surface2"]};border:1px solid {T["border"]};padding:10px 14px;margin-bottom:8px;}}

/* ── STAT BLOCK ── */
.stat-block{{text-align:center;padding:12px 8px;border:1px solid {T["border"]};background:{T["surface"]};}}
.stat-num{{font-family:'Playfair Display',serif;font-size:36px;font-weight:700;line-height:1;color:{T["ink"]};}}
.stat-label{{font-family:'IBM Plex Mono',monospace;font-size:8px;color:{T["ink3"]};
  letter-spacing:.12em;text-transform:uppercase;margin-top:4px;}}

/* ── PULL QUOTE ── */
.pull-quote{{font-family:'Playfair Display',serif;font-size:18px;font-style:italic;
  line-height:1.45;color:{T["ink"]};text-align:center;
  border-top:2px solid {T["ink"]};border-bottom:1px solid {T["border"]};
  padding:12px 8px;margin:10px 0;}}

/* ── STRATEGY TIMELINE ── */
.tl-item{{display:grid;grid-template-columns:56px 1fr;gap:12px;
  padding:10px 0;border-bottom:1px solid {T["border"]};}}
.tl-item:last-child{{border-bottom:none;}}
.tl-week{{font-family:'IBM Plex Mono',monospace;font-size:8px;color:{T["accent"]};
  letter-spacing:.08em;text-transform:uppercase;line-height:1.4;padding-top:2px;cursor:pointer;}}
.tl-week:hover{{color:{T["accent2"]};}}
.tl-text{{font-size:11px;color:{T["ink2"]};line-height:1.65;}}
.tl-text strong{{color:{T["ink"]};font-weight:600;}}
.tl-active{{background:{T["surface2"]};margin:-10px -14px;padding:10px 14px;border-left:3px solid {T["accent"]};}}

/* ── NLP STATS ── */
.nlp-row{{display:flex;align-items:baseline;gap:8px;margin-bottom:12px;
  padding-bottom:12px;border-bottom:1px solid {T["border"]};}}
.nlp-row:last-child{{border-bottom:none;margin-bottom:0;padding-bottom:0;}}
.nlp-num{{font-family:'Playfair Display',serif;font-size:28px;font-weight:700;color:{T["ink"]};}}
.nlp-label{{font-size:11px;color:{T["ink2"]};}}
.nlp-tag{{font-family:'IBM Plex Mono',monospace;font-size:9px;padding:2px 7px;
  border:1px solid {T["border2"]};color:{T["ink3"]};margin-left:auto;}}
.nlp-tag.pos{{border-color:{T["gold"]};color:{T["gold"]};}}
.nlp-tag.neg{{border-color:{T["accent"]};color:{T["accent"]};}}

/* ── CHAT ── */
.chat-section-head{{background:{T["ink"]};color:{T["surface"]};
  font-family:'Playfair Display',serif;font-size:16px;font-weight:700;
  padding:10px 20px;border-bottom:2px solid {T["accent"]};}}
.chat-messages{{padding:16px 20px;display:flex;flex-direction:column;gap:8px;min-height:120px;}}
.chat-msg{{font-size:12px;line-height:1.65;padding:8px 12px;max-width:85%;}}
.chat-msg.user{{background:{T["ink"]};color:{T["surface"]};align-self:flex-end;font-style:italic;}}
.chat-msg.ai{{background:{T["surface2"]};border:1px solid {T["border"]};
  align-self:flex-start;color:{T["ink2"]};}}
.chat-sugg-row{{display:flex;gap:6px;flex-wrap:wrap;padding:0 20px 12px;}}
.chat-sugg{{font-family:'IBM Plex Mono',monospace;font-size:9px;color:{T["ink2"]};
  padding:4px 10px;border:1px solid {T["border2"]};cursor:pointer;
  background:{T["surface"]};transition:all .15s;}}
.chat-sugg:hover{{background:{T["ink"]};color:{T["surface"]};border-color:{T["ink"]};}}

/* ── FORM ELEMENTS ── */
[data-testid="stSelectbox"]>div>div{{background:{T["input_bg"]}!important;border:1px solid {T["input_border"]}!important;border-radius:0!important;color:{T["input_color"]}!important;font-size:12px!important;font-family:'IBM Plex Sans',sans-serif!important;}}
[data-testid="stNumberInput"]>div>div>input{{background:{T["input_bg"]}!important;border:1px solid {T["input_border"]}!important;color:{T["input_color"]}!important;border-radius:0!important;font-size:12px!important;font-family:'IBM Plex Mono',monospace!important;}}
textarea{{background:{T["input_bg"]}!important;border:1px solid {T["input_border"]}!important;color:{T["input_color"]}!important;border-radius:0!important;font-size:12px!important;font-family:'IBM Plex Sans',sans-serif!important;}}
[data-testid="stSlider"]>div>div>div>div{{background:{T["accent"]}!important;}}
[data-testid="stSlider"]>div>div>div{{background:{T["border2"]}!important;}}
.stSlider label{{font-size:11px!important;color:{T["ink2"]}!important;font-family:'IBM Plex Mono',monospace!important;}}
.stButton>button[kind="primary"]{{background:{T["ink"]}!important;border:none!important;border-radius:0!important;
  font-family:'IBM Plex Mono',monospace!important;font-weight:500!important;font-size:11px!important;
  letter-spacing:.1em!important;color:{T["surface"]}!important;text-transform:uppercase!important;
  border-bottom:2px solid {T["accent"]}!important;}}
.stButton>button[kind="primary"]:hover{{background:{T["accent"]}!important;}}
.stButton>button[kind="secondary"]{{background:transparent!important;border:1px solid {T["border2"]}!important;
  border-radius:0!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;
  letter-spacing:.08em!important;color:{T["ink2"]}!important;}}
.stButton>button[kind="secondary"]:hover{{border-color:{T["ink"]}!important;color:{T["ink"]}!important;}}
[data-testid="stFileUploader"] section{{background:{T["surface2"]}!important;border:1px dashed {T["border2"]}!important;border-radius:0!important;}}
[data-testid="stCheckbox"] label{{font-size:12px!important;color:{T["ink2"]}!important;font-family:'IBM Plex Mono',monospace!important;}}
[data-testid="stToggle"] label{{font-size:12px!important;color:{T["ink2"]}!important;}}
audio{{width:100%!important;height:32px!important;border-radius:0!important;accent-color:{T["accent"]};}}
[data-testid="stProgress"]>div>div{{background:{T["accent"]}!important;}}
[data-testid="stProgress"]>div{{background:{T["border2"]}!important;height:2px!important;}}
[data-testid="stTabs"]>div:first-child{{border-bottom:1px solid {T["border"]}!important;}}
[data-testid="stTabs"] button{{font-family:'IBM Plex Mono',monospace!important;font-size:9px!important;font-weight:500!important;color:{T["ink3"]}!important;letter-spacing:.1em!important;text-transform:uppercase!important;padding:8px 16px!important;border-radius:0!important;background:transparent!important;}}
[data-testid="stTabs"] button[aria-selected="true"]{{color:{T["accent"]}!important;border-bottom:2px solid {T["accent"]}!important;}}
[data-testid="stAlert"]{{border-radius:0!important;font-size:12px!important;}}
[data-testid="stExpander"]{{border-radius:0!important;border:1px solid {T["border"]}!important;background:{T["surface"]}!important;}}

/* Hover tooltip for feature bars */
.feat-tooltip{{position:relative;cursor:help;}}
.feat-tooltip:hover .feat-tooltip-text{{display:block;}}
.feat-tooltip-text{{display:none;position:absolute;left:50%;transform:translateX(-50%);
  bottom:calc(100% + 6px);background:{T["ink"]};color:{T["surface"]};
  font-family:'IBM Plex Sans',sans-serif;font-size:10px;line-height:1.5;
  padding:6px 10px;white-space:nowrap;z-index:99;border-left:2px solid {T["accent"]};}}
</style>""", unsafe_allow_html=True)

# ── TOPBAR ──────────────────────────────────────────────────
done_count = sum([st.session_state.auto_detected,
                  st.session_state.result is not None,
                  st.session_state.result is not None,
                  st.session_state.result is not None])

st.markdown(f'''<div class="topbar">
  <div class="topbar-logo">Stream<span>Breaker</span> AI</div>
  <div class="topbar-meta">
    Pipeline: {done_count}/4 &nbsp;·&nbsp;
    {"✓ Analysis complete" if done_count==4 else "Awaiting analysis"}
  </div>
</div>
<div class="dateline">StreamBreaker AI &nbsp;·&nbsp; Music Marketing Intelligence &nbsp;·&nbsp; Models: XGBoost · TextBlob · GPT-3.5 · librosa</div>
''', unsafe_allow_html=True)

# ── NAV BUTTONS ─────────────────────────────────────────────
n0,n1,n2,n3,n4,n5 = st.columns([1.2,1,1,1,0.7,0.7])
with n0:
    st.markdown(f'<div style="padding:8px 40px;font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};letter-spacing:.1em;text-transform:uppercase;">Navigate</div>', unsafe_allow_html=True)
for col,pid,label in [(n1,"analyze","🎵  Analyze"),(n2,"report","📋  Report"),(n3,"settings","⚙️  Settings")]:
    with col:
        active = page == pid
        if st.button(label, key=f"nav_{pid}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = pid; st.rerun()
with n4:
    st.markdown(f'<div style="padding:6px 0;text-align:center;font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};letter-spacing:.08em;">{'✓ COMPLETE' if done_count==4 else f'{done_count}/4 done'}</div>', unsafe_allow_html=True)
with n5:
    if st.button(T["tog"], key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode; st.rerun()

st.markdown(f'<div style="height:1px;background:{T["border"]};"></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: ANALYZE
# ═══════════════════════════════════════════════════════════
if page == "analyze":
    st.markdown('<div class="page-pad">', unsafe_allow_html=True)

    # Determine API key
    api_key_input = (st.session_state.openai_key or
                     os.getenv("OPENAI_API_KEY",""))
    try:
        if not api_key_input: api_key_input = st.secrets.get("OPENAI_API_KEY","")
    except: pass

    col_l, col_r = st.columns([1, 1.8], gap="large")

    with col_l:
        st.markdown(f'''
        <div style="padding-top:28px;">
          <div class="ed-kicker">Track Input</div>
          <div class="ed-title-sm">Configure Your Track</div>
          <div class="ed-byline">Upload · Set Features · Add Lyrics · Run</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        # LLM selector
        st.markdown(f'<div class="sec-head">LLM Engine <span>Model 3</span></div>', unsafe_allow_html=True)
        llm_choice = st.selectbox("LLM", ["OpenAI","Claude (Anthropic)","Groq (Fast & Free)","Ollama (Local)"], label_visibility="collapsed")
        backend_map = {"OpenAI":"openai","Claude (Anthropic)":"claude","Groq (Fast & Free)":"groq","Ollama (Local)":"ollama"}
        selected_backend = backend_map[llm_choice]
        if selected_backend == "claude":
            api_key_input = st.session_state.claude_key or os.getenv("ANTHROPIC_API_KEY","")
        elif selected_backend == "groq":
            api_key_input = st.session_state.groq_key or os.getenv("GROQ_API_KEY","")
        if not api_key_input and selected_backend != "ollama":
            st.warning("No API key — add one in ⚙️ Settings")

        # Upload
        st.markdown(f'<div class="sec-head" style="margin-top:16px;">Track Upload <span>Model 0</span></div>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload", type=["mp3","wav","m4a","flac","ogg"], label_visibility="collapsed")

        auto_vals = {}
        if audio_file:
            fname = re.sub(r'[\[\(]?\d{5,}[\]\)]?', '', audio_file.name.rsplit('.',1)[0].replace('_',' ')).strip()
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:{T["ink3"]};margin:6px 0 4px;letter-spacing:.06em;">▶ NOW PLAYING</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:12px;font-weight:500;color:{T["ink2"]};margin-bottom:6px;">{fname}</div>', unsafe_allow_html=True)
            audio_file.seek(0)
            st.audio(audio_file, format=f"audio/{audio_file.name.rsplit('.',1)[-1].lower()}")
            audio_file.seek(0)

            with st.spinner("Analyzing full track…"):
                meta = get_file_metadata(audio_file, filename=audio_file.name)
                extracted, err = extract_audio_features(audio_file, filename=audio_file.name)
                embedded = meta.get("embedded_lyrics","")
                if embedded and len(embedded.strip())>20:
                    lyr_text, lyr_source = embedded.strip(), "embedded"
                elif api_key_input and selected_backend=="openai":
                    lyr_text, lyr_source = fetch_lyrics_gpt(meta["title"],meta["artist"],api_key_input)
                else:
                    lyr_text, lyr_source = "", "not_found"
                st.session_state.auto_lyrics = lyr_text
                st.session_state.lyrics_source = lyr_source
            if err:
                st.error(err)
            else:
                auto_vals = extracted
                st.session_state.auto_detected = True
                lyr_badge = {"embedded":"✓ Lyrics embedded","gpt":"✓ Lyrics via GPT","not_found":"✏ Paste lyrics below"}.get(lyr_source,"")
                st.markdown(f'''<div style="background:{T["surface2"]};border-left:3px solid {T["gold"]};
                  padding:8px 12px;font-family:IBM Plex Mono,monospace;font-size:9px;
                  color:{T["gold"]};letter-spacing:.06em;margin:8px 0;">
                  ✓ {extracted["_key_name"]} {extracted["_mode_name"]} · {extracted["tempo"]:.0f} BPM · {lyr_badge}
                </div>''', unsafe_allow_html=True)
        else:
            st.session_state.auto_detected = False
            st.session_state.auto_lyrics = ""
            st.session_state.lyrics_source = ""

        disabled = bool(auto_vals) and not st.toggle("Override auto-detected values", value=not bool(auto_vals))

        # Audio features
        st.markdown(f'<div class="sec-head" style="margin-top:14px;">Audio Features <span>Model 0 · librosa</span></div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            danceability = st.slider("Danceability",0.0,1.0,float(auto_vals.get("danceability",0.65)),0.01,disabled=disabled)
            energy       = st.slider("Energy",0.0,1.0,float(auto_vals.get("energy",0.75)),0.01,disabled=disabled)
            valence      = st.slider("Valence",0.0,1.0,float(auto_vals.get("valence",0.55)),0.01,disabled=disabled)
            acousticness = st.slider("Acousticness",0.0,1.0,float(auto_vals.get("acousticness",0.15)),0.01,disabled=disabled)
        with c2:
            speechiness      = st.slider("Speechiness",0.0,1.0,float(auto_vals.get("speechiness",0.05)),0.01,disabled=disabled)
            instrumentalness = st.slider("Instrumental.",0.0,1.0,float(auto_vals.get("instrumentalness",0.0)),0.01,disabled=disabled)
            liveness         = st.slider("Liveness",0.0,1.0,float(auto_vals.get("liveness",0.12)),0.01,disabled=disabled)
            loudness         = st.slider("Loudness (dB)",-60.0,0.0,float(auto_vals.get("loudness",-6.0)),0.5,disabled=disabled)

        tempo        = st.slider("Tempo (BPM)",60,220,int(auto_vals.get("tempo",125)),1,disabled=disabled)
        duration_sec = st.slider("Duration (sec)",30,600,int(auto_vals.get("duration_ms",210000)/1000),5)

        c3,c4 = st.columns(2)
        with c3:
            key     = st.selectbox("Key",list(range(12)),index=int(auto_vals.get("key",5)),format_func=lambda x:KEY_NAMES[x],disabled=disabled)
            time_sig= st.selectbox("Time Sig",[3,4,5],index=1)
        with c4:
            mode    = st.selectbox("Mode",[0,1],index=int(auto_vals.get("mode",1)),format_func=lambda x:"Major" if x==1 else "Minor",disabled=disabled)
            explicit= st.checkbox("Explicit",value=False)

        genre = st.selectbox("Genre",["indie","indie-pop","indie-rock","indie-folk","folk","acoustic",
            "alternative","singer-songwriter","dream-pop","lo-fi","pop","rock","electronic","hip-hop","r-n-b"],index=1)

        # Profile from Settings
        career_stage      = st.session_state.career_val
        instagram         = st.session_state.instagram_val
        spotify_listeners = st.session_state.spotify_val
        youtube_subs      = st.session_state.youtube_val
        budget            = st.session_state.budget_val

        st.markdown(f'''<div style="display:flex;align-items:center;justify-content:space-between;
          border:1px solid {T["border"]};padding:8px 12px;margin:10px 0;background:{T["surface2"]};">
          <div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink2"]};letter-spacing:.06em;">
            {career_stage.upper()} · ${budget:,} BUDGET
          </div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};">edit in ⚙ settings</div>
        </div>''', unsafe_allow_html=True)

        # Lyrics
        st.markdown(f'<div class="sec-head" style="margin-top:14px;">Song Lyrics <span>Model 2 · NLP</span></div>', unsafe_allow_html=True)
        ls = st.session_state.lyrics_source
        if ls == "not_found":
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};margin-bottom:4px;letter-spacing:.06em;">NOT FOUND AUTOMATICALLY — PASTE BELOW</div>', unsafe_allow_html=True)
        elif ls in ("gpt","embedded"):
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["gold"]};margin-bottom:4px;letter-spacing:.06em;">{"✓ VIA GPT" if ls=="gpt" else "✓ FROM FILE"} — EDIT IF NEEDED</div>', unsafe_allow_html=True)
        lyrics = st.text_area("Lyrics", height=130, label_visibility="collapsed",
            value=st.session_state.auto_lyrics,
            placeholder="[Verse 1]\nPaste lyrics here...\n\n[Chorus]\nSection headers improve NLP analysis.")

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        run_btn = st.button("⚡  Run Analysis", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div style="padding-top:28px;">', unsafe_allow_html=True)
        if st.session_state.result is None:
            st.markdown(f'''<div style="display:flex;flex-direction:column;align-items:center;
              justify-content:center;min-height:70vh;text-align:center;padding:40px;">
              <div style="font-family:Playfair Display,serif;font-size:64px;font-style:italic;
                   color:{T["border2"]};line-height:1;margin-bottom:20px;">Ready.</div>
              <div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink4"]};
                   letter-spacing:.14em;text-transform:uppercase;margin-bottom:24px;">
                Configure · Upload · Analyze
              </div>
              <div style="display:flex;gap:16px;align-items:center;font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink4"]};letter-spacing:.08em;">
                <span>M0 Audio</span><span>→</span><span>M1 XGBoost</span><span>→</span><span>M2 NLP</span><span>→</span><span>M3 Strategy</span>
              </div>
            </div>''', unsafe_allow_html=True)
        else:
            result = st.session_state.result
            af     = st.session_state.af
            pred   = result["model1_prediction"]
            nlp    = result["model2_nlp"]
            strat  = result["model3_strategy"]
            pp     = pred["prediction_probability"]
            st.markdown(f'''<div style="border-bottom:1px solid {T["border"]};padding-bottom:12px;margin-bottom:16px;">
              <div class="ed-kicker">Current Analysis</div>
              <div style="font-family:Playfair Display,serif;font-size:28px;font-weight:700;color:{T["ink"]};">
                {pp:.2f}% Stream Score
              </div>
              <div style="font-size:12px;color:{T["ink2"]};font-style:italic;margin-top:4px;">
                {"High potential — invest in marketing" if pp>=70 else "Moderate potential — targeted spend" if pp>=50 else "Low potential — refine before release"}
              </div>
            </div>''', unsafe_allow_html=True)
            # Quick summary cards
            ks = st.columns(3)
            for col_k,val,lbl in [
                (ks[0],f"{pp:.0f}%","Stream Score"),
                (ks[1],f"{nlp['hook_repetition']:.0%}","Hook Score"),
                (ks[2],nlp["sentiment"].title(),"Sentiment"),
            ]:
                with col_k:
                    st.markdown(f'''<div class="stat-block">
                      <div class="stat-num">{val}</div>
                      <div class="stat-label">{lbl}</div>
                    </div>''', unsafe_allow_html=True)
            st.markdown(f'''<div style="margin-top:12px;padding:10px 14px;background:{T["surface2"]};border-left:3px solid {T["accent"]};
              font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};letter-spacing:.08em;">
              VIEW FULL REPORT → click <strong style="color:{T["accent"]};">📋 Report</strong> in the navigation
            </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RUN
    if run_btn:
        af_input = {"danceability":danceability,"energy":energy,"key":key,"loudness":loudness,
                    "mode":mode,"speechiness":speechiness,"acousticness":acousticness,
                    "instrumentalness":instrumentalness,"liveness":liveness,"valence":valence,
                    "tempo":float(tempo),"duration_ms":duration_sec*1000,"time_signature":time_sig,
                    "explicit":explicit,"genre":genre}
        artist_profile = {"instagram_followers":instagram,"spotify_listeners":spotify_listeners,
                          "youtube_subscribers":youtube_subs,"genre":genre.replace("-"," ").title()}
        st.session_state["budget_val"] = budget
        st.session_state["_run_af"]    = af_input
        pipeline = load_pipeline(backend=selected_backend, api_key=api_key_input or None)
        prog = st.progress(0, text="Initializing…")
        prog.progress(15, text="📊 Model 1 · XGBoost…")
        try:
            prog.progress(45, text="🎤 Model 2 · NLP…")
            prog.progress(72, text="🚀 Model 3 · Strategy…")
            result = pipeline.run(audio_features=af_input, lyrics=lyrics, budget=budget,
                                  artist_profile=artist_profile, career_stage=career_stage)
            prog.progress(100, text="✅ Done"); prog.empty()
            st.session_state.result = result
            st.session_state.af     = af_input
            st.rerun()
        except Exception as e:
            prog.empty(); st.error(f"Pipeline error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: REPORT
# ═══════════════════════════════════════════════════════════
elif page == "report":
    st.markdown('<div class="page-pad">', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown(f'''<div style="text-align:center;padding:80px 40px;">
          <div style="font-family:Playfair Display,serif;font-size:48px;font-style:italic;color:{T["border2"]};">No Analysis Yet</div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink4"]};letter-spacing:.14em;text-transform:uppercase;margin-top:12px;">
            Return to Analyze → configure your track → run analysis
          </div>
        </div>''', unsafe_allow_html=True)
    else:
        result  = st.session_state.result
        af      = st.session_state.af
        pred    = result["model1_prediction"]
        nlp     = result["model2_nlp"]
        strat   = result["model3_strategy"]
        pp      = pred["prediction_probability"]
        budget  = int(st.session_state.budget_val)
        genre   = st.session_state.get("_run_genre","indie-pop")

        # ── REPORT MASTHEAD ──────────────────────────────
        verdict = "HIGH POTENTIAL" if pp>=70 else "MODERATE POTENTIAL" if pp>=50 else "LOW POTENTIAL"
        deck_text = (
            f"A track with an {nlp['hook_repetition']:.0%} hook score and {nlp['sentiment']} sentiment — "
            f"{'exceptional viral indicators' if nlp['hook_repetition']>=0.7 else 'moderate viral potential'} — "
            f"{'held back by early-stage audience metrics' if pp<70 else 'ready for full marketing investment'}."
        )
        dur_str = f"{af['duration_ms']//1000//60}:{(af['duration_ms']//1000)%60:02d}"
        key_str = f"{KEY_NAMES[af['key']]} {'Major' if af['mode']==1 else 'Minor'}"

        col_title, col_score = st.columns([2, 0.8], gap="medium")
        with col_title:
            st.markdown(f'''
            <div style="padding-top:28px;padding-bottom:20px;border-bottom:3px double {T["border2"]};">
              <div class="ed-kicker">Analysis Report</div>
              <div class="ed-title">{genre.replace("-"," ").title()} Track</div>
              <div class="ed-deck">{deck_text}</div>
              <div class="ed-byline">
                {key_str} &nbsp;·&nbsp; {af["tempo"]:.0f} BPM &nbsp;·&nbsp; {dur_str} &nbsp;·&nbsp;
                {st.session_state.career_val.title()} Artist &nbsp;·&nbsp; ${budget:,} Budget
              </div>
            </div>
            ''', unsafe_allow_html=True)
        with col_score:
            sc = "#8a8060" if pp<70 else T["gold"]
            st.markdown(f'''<div style="padding-top:28px;padding-bottom:20px;border-bottom:3px double {T["border2"]};text-align:right;">
              <div class="ed-kicker">Streaming Probability</div>
              <div style="font-family:Playfair Display,serif;font-size:64px;font-weight:900;
                   color:{sc};line-height:.9;letter-spacing:-.02em;">{pp:.0f}%</div>
              <div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};
                   letter-spacing:.1em;margin-top:6px;">{verdict}</div>
              <div style="margin-top:8px;display:inline-block;border:1px solid {sc};
                   color:{sc};font-family:IBM Plex Mono,monospace;font-size:9px;
                   letter-spacing:.1em;padding:4px 12px;">
                {"✓ WILL HIT 1K STREAMS" if pred["will_hit_1k_streams"] else "✗ BELOW 1K THRESHOLD"}
              </div>
            </div>''', unsafe_allow_html=True)

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # ── THREE-COLUMN NEWSPAPER ────────────────────────
        nc1_w, div1_w, nc2_w, div2_w, nc3_w = 2.2, 0.05, 2.8, 0.05, 2.0
        nc1, _d1, nc2, _d2, nc3 = st.columns([nc1_w, div1_w, nc2_w, div2_w, nc3_w], gap="small")

        # COL 1: Score gauge + audio features
        with nc1:
            st.markdown(f'<div class="sec-head">Prediction · M1 <span>XGBoost</span></div>', unsafe_allow_html=True)
            st.markdown('<div style="display:flex;justify-content:center;padding:8px 0;">', unsafe_allow_html=True)
            st.markdown(gauge_editorial(pp), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            conf_color = T["gold"] if pred["confidence"]=="high" else T["ink2"]
            st.markdown(f'''<div style="text-align:center;font-family:IBM Plex Mono,monospace;
              font-size:9px;color:{conf_color};letter-spacing:.08em;margin-bottom:14px;">
              {pred["confidence"].upper()} CONFIDENCE
            </div>''', unsafe_allow_html=True)

            st.markdown(f'<div class="sec-head" style="margin-top:4px;">Audio Features <span>▸ hover for details</span></div>', unsafe_allow_html=True)

            # Feature bars with inline tooltips
            FEAT_INFO = {
                "Danceability": ("0.65","Tempo regularity + beat strength. Higher = more rhythmically consistent."),
                "Energy":       ("0.70","Perceptual intensity — loudness + onset strength. Above avg for indie-pop."),
                "Valence":      ("0.50","Musical positiveness. Higher = more upbeat and happy-sounding."),
                "Acousticness": ("0.25","Acoustic vs electronic. Low = more produced/electronic sound."),
                "Liveness":     ("0.15","Audience presence probability. Low = studio recording."),
                "Speechiness":  ("0.05","Speech-like content. Very low = pure music, no rap/talk."),
            }
            feats = [
                ("Danceability",af["danceability"]),("Energy",af["energy"]),
                ("Valence",af["valence"]),("Acousticness",af["acousticness"]),
                ("Liveness",af["liveness"]),("Speechiness",af["speechiness"]),
            ]
            for fname, fval in feats:
                bench_str, tip = FEAT_INFO.get(fname, ("0.50",""))
                bench_f = float(bench_str)
                bar_pct = int(min(100, max(0, fval*100)))
                bench_pct = int(bench_f*100)
                clr = T["gold"] if fval >= bench_f else T["accent"]
                above = "above" if fval >= bench_f else "below"
                st.markdown(f'''<div class="feat-tooltip" style="margin-bottom:10px;">
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};min-width:90px;">{fname}</span>
                    <div style="flex:1;height:2px;background:{T["border2"]};position:relative;">
                      <div style="width:{bar_pct}%;height:2px;background:{clr};"></div>
                      <div style="position:absolute;left:{bench_pct}%;top:-3px;width:1px;height:8px;background:{T["ink4"]};"></div>
                    </div>
                    <span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{clr};min-width:28px;text-align:right;">{fval:.2f}</span>
                  </div>
                  <div class="feat-tooltip-text">{tip}<br><em>{fval:.2f} — {above} genre avg ({bench_str})</em></div>
                </div>''', unsafe_allow_html=True)

            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["ink4"]};margin-top:4px;">│ = indie-pop genre average</div>', unsafe_allow_html=True)

        # Divider
        with _d1:
            st.markdown(f'<div style="margin-top:0;width:1px;background:{T["border"]};min-height:500px;"></div>', unsafe_allow_html=True)

        # COL 2: Strategy timeline + projection
        with nc2:
            st.markdown(f'<div class="sec-head">Marketing Strategy · M3 <span>GPT-3.5</span></div>', unsafe_allow_html=True)

            # Week selector
            wk_cols = st.columns(4)
            for wi, wlbl in enumerate(["Week 1","Week 2","Week 3","Week 4"]):
                with wk_cols[wi]:
                    is_active = st.session_state.active_week == wi
                    if st.button(wlbl, key=f"wk_{wi}",
                                 type="primary" if is_active else "secondary",
                                 use_container_width=True):
                        st.session_state.active_week = wi if not is_active else None
                        st.rerun()

            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

            week_details = [
                ("Week 1","Launch",
                 "Front-load Meta Ads at <strong>$150/day</strong> Days 1–2, then <strong>$85/day</strong> Days 3–6. "
                 "Critical Day 6 checkpoint: achieve <strong>40%+ save rate (240+ saves)</strong> to trigger "
                 "Spotify's algorithmic testing on Day 7. Target US, Canada, Brazil geo-mix."),
                ("Week 2","Algorithm",
                 "Scale back to <strong>$30/day</strong> maintenance. Monitor Release Radar expansion. "
                 "Track streams-per-listener ratio — target <strong>&gt;2.0</strong>. "
                 "Watch for playlist add notifications from SubmitHub pitches."),
                ("Week 3","TikTok Push",
                 "Deploy creator partnerships: <strong>$300</strong> for 2–4 micro-influencers (10K–50K followers). "
                 f"Your hook score of <strong>{nlp['hook_repetition']:.0%}</strong> makes the chorus an ideal "
                 "15-second clip. Organic challenge content: $150."),
                ("Week 4","Evaluate",
                 "Compare actual cost-per-stream against <strong>$0.022 indie-pop benchmark</strong>. "
                 "Calculate ROI. Assess Discover Weekly probability with 10K+ streams + 10% save rate. "
                 "Document learnings for next release cycle."),
            ]
            for wi, (wk, subtitle, text) in enumerate(week_details):
                is_active = st.session_state.active_week == wi
                active_style = f"background:{T['surface2']};border-left:3px solid {T['accent']};margin-left:-3px;" if is_active else ""
                st.markdown(f'''<div style="display:grid;grid-template-columns:52px 1fr;gap:12px;
                  padding:10px 12px {active_style};border-bottom:1px solid {T["border"]};">
                  <div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["accent"]};
                         letter-spacing:.08em;text-transform:uppercase;">{wk}</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:7px;color:{T["ink3"]};
                         letter-spacing:.06em;text-transform:uppercase;margin-top:2px;">{subtitle}</div>
                  </div>
                  <div style="font-size:11px;color:{T["ink2"] if not is_active else T["ink"]};line-height:1.65;">{text}</div>
                </div>''', unsafe_allow_html=True)

            if is_active := (st.session_state.active_week is not None):
                active_idx = st.session_state.active_week
                st.markdown(f'''<div style="background:{T["surface2"]};border:1px solid {T["border"]};
                  padding:8px 12px;margin-top:4px;">
                  <div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["accent"]};
                       letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;">
                    {week_details[active_idx][0]} · Key Metrics
                  </div>
                  <div style="font-size:10px;color:{T["ink2"]};line-height:1.65;">
                    {["Target: 240 saves by Day 6 · $600 Meta Ads · 38 playlist pitches",
                       "Target: streams/listener > 2.0 · Release Radar expansion · $30/day maintenance",
                       "Target: 3k–6k TikTok streams · 2–4 creator posts · high hook = strong clip",
                       f"Target: {int(budget/0.022*(pp/100)*0.7):,}–{int(budget/0.022*(pp/100)*1.3):,} total streams · CPS ≤ $0.022"][active_idx]}
                  </div>
                </div>''', unsafe_allow_html=True)

            # Projection chart
            st.markdown(f'<div class="sec-head" style="margin-top:20px;">Stream Projection <span>4 weeks · $0.022 CPS</span></div>', unsafe_allow_html=True)
            st.markdown(projection_svg_ed(pp, budget), unsafe_allow_html=True)
            lo = int(budget/0.022*(pp/100)*0.7); hi = int(budget/0.022*(pp/100)*1.3)
            st.markdown(f'''<div style="display:flex;justify-content:center;gap:32px;margin-top:6px;">
              <div style="text-align:center;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:14px;font-weight:500;color:{T["ink2"]};">{lo:,}</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:7px;color:{T["ink4"]};letter-spacing:.1em;text-transform:uppercase;">Conservative</div>
              </div>
              <div style="text-align:center;">
                <div style="font-family:IBM Plex Mono,monospace;font-size:14px;font-weight:500;color:{T["accent"]};">{hi:,}</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:7px;color:{T["ink4"]};letter-spacing:.1em;text-transform:uppercase;">Optimistic</div>
              </div>
            </div>''', unsafe_allow_html=True)

        # Divider
        with _d2:
            st.markdown(f'<div style="margin-top:0;width:1px;background:{T["border"]};min-height:500px;"></div>', unsafe_allow_html=True)

        # COL 3: NLP + Budget
        with nc3:
            st.markdown(f'<div class="sec-head">Lyric Analysis · M2 <span>TextBlob</span></div>', unsafe_allow_html=True)

            hook_v  = nlp["hook_repetition"]
            lex_v   = nlp["lexical_diversity"]
            coh_v   = nlp["semantic_coherence"]
            sent    = nlp["sentiment"]

            # Pull quote style NLP
            hook_verdict = "High Viral Potential" if hook_v>=0.7 else "Moderate Hook" if hook_v>=0.4 else "Weak Hook"
            st.markdown(f'<div class="pull-quote">{hook_verdict}</div>', unsafe_allow_html=True)

            nlp_items = [
                (f"{hook_v:.0%}",   "Hook Repetition", "pos" if hook_v>=0.6 else "neg" if hook_v<0.4 else "", "🔥 High viral" if hook_v>=0.7 else "Moderate" if hook_v>=0.4 else "Weak"),
                (sent.title(),       "Sentiment",        "pos" if sent=="positive" else "neg" if sent=="negative" else "", "Broad appeal" if sent=="positive" else "Niche" if sent=="negative" else "Neutral"),
                (f"{lex_v:.2f}",    "Lexical Diversity","pos" if lex_v>=0.7 else "", "Rich" if lex_v>=0.7 else "Moderate" if lex_v>=0.4 else "Simple"),
                (f"{coh_v:.2f}",    "Coherence",        "pos" if coh_v>=0.6 else "", "Strong theme" if coh_v>=0.6 else "Mixed"),
                ("Clean" if not nlp["profanity_detected"] else "Explicit",
                 "Content",         "pos" if not nlp["profanity_detected"] else "neg", "Playlist eligible" if not nlp["profanity_detected"] else "Restricted"),
            ]
            for nval,nlbl,ntag_cls,ntag_text in nlp_items:
                st.markdown(f'''<div class="nlp-row">
                  <div style="display:flex;align-items:baseline;gap:8px;flex:1;">
                    <span class="nlp-num" style="font-size:{"20px" if len(nval)>4 else "24px"};">{nval}</span>
                    <span class="nlp-label">{nlbl}</span>
                  </div>
                  <span class="nlp-tag {ntag_cls}">{ntag_text}</span>
                </div>''', unsafe_allow_html=True)

            # Budget
            st.markdown(f'<div class="sec-head" style="margin-top:16px;">Budget · M3 <span>${budget:,}</span></div>', unsafe_allow_html=True)
            st.markdown(budget_donut_svg(budget), unsafe_allow_html=True)

            # Interactive budget adjuster
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["ink3"]};letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;">Adjust Budget (updates projection)</div>', unsafe_allow_html=True)
            new_budget = st.slider("Budget adj", 200, 10000, budget, 100, label_visibility="collapsed", key="report_budget_slider")
            if new_budget != budget:
                st.session_state.budget_val = new_budget
                st.rerun()

        # ── FULL-WIDTH DIVIDER ───────────────────────────
        st.markdown(f'<div style="height:1px;background:{T["border2"]};margin:24px 0;"></div>', unsafe_allow_html=True)

        # ── STRATEGY FULL TEXT (collapsible) ────────────
        if strat and strat.get("success"):
            with st.expander("📄 Full GPT-3.5 Marketing Strategy Report", expanded=False):
                ms = strat["metadata"]
                st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};margin-bottom:12px;letter-spacing:.08em;">{ms["model"]} · {ms["tokens_used"]} tokens · ${ms["cost_estimate"]:.4f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:13px;line-height:1.85;color:{T["ink2"]};">{strat["strategy"]}</div>', unsafe_allow_html=True)

        # ── CHAT SECTION ─────────────────────────────────
        st.markdown(f'<div style="border-top:2px solid {T["ink"]};margin-top:4px;">', unsafe_allow_html=True)
        st.markdown(f'''<div class="chat-section-head">
          Ask the Analyst &nbsp;·&nbsp;
          <span style="font-family:IBM Plex Mono,monospace;font-size:10px;font-weight:400;
            color:{T["accent"]};">Full analysis context loaded</span>
        </div>''', unsafe_allow_html=True)

        chat_key = (st.session_state.claude_key or
                    st.session_state.openai_key or
                    os.getenv("OPENAI_API_KEY",""))
        try:
            if not chat_key: chat_key = st.secrets.get("OPENAI_API_KEY","")
        except: pass

        system_prompt = (
            "You are StreamBreaker AI's music marketing analyst. You have the artist's full analysis:\n"
            f"Stream Score: {pp:.2f}% ({pred['confidence']} confidence) · "
            f"Will hit 1K: {pred['will_hit_1k_streams']} · "
            f"Hook: {nlp['hook_repetition']:.0%} · Sentiment: {nlp['sentiment']} · "
            f"Lexical diversity: {nlp['lexical_diversity']:.2f} · Coherence: {nlp['semantic_coherence']:.2f} · "
            f"Energy: {af['energy']:.2f} · Danceability: {af['danceability']:.2f} · "
            f"Valence: {af['valence']:.2f} · Tempo: {af['tempo']:.0f}BPM · "
            f"Key: {KEY_NAMES[af['key']]} {'Major' if af['mode']==1 else 'Minor'} · "
            f"Budget: ${budget:,} · Career: {st.session_state.career_val} · "
            f"Spotify listeners: {st.session_state.spotify_val:,}\n\n"
            "Be direct, specific, and reference the actual numbers. Keep answers concise and actionable."
        )

        chat_layout_l, chat_layout_r = st.columns([2.5, 1])

        with chat_layout_l:
            # Display history
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
            if not st.session_state.chat_history:
                st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink4"]};letter-spacing:.08em;padding:8px 0;">Ask anything about your analysis, scores, strategy, or release timing.</div>', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                role = msg["role"]; text = msg["content"]
                st.markdown(f'<div class="chat-msg {role}">{text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Process pending AI response
            if (st.session_state.chat_history and
                st.session_state.chat_history[-1]["role"] == "user"):
                if not chat_key:
                    st.warning("Add an API key in ⚙️ Settings to use the chat.")
                else:
                    with st.spinner(""):
                        try:
                            msgs = [{"role":m["role"],"content":m["content"]}
                                    for m in st.session_state.chat_history]
                            reply = chat_call(msgs, system_prompt)
                            st.session_state.chat_history.append({"role":"assistant","content":reply})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Chat error: {str(e)[:100]}")

            user_input = st.chat_input("Ask about your track, scores, strategy, or next steps…")
            if user_input:
                st.session_state.chat_history.append({"role":"user","content":user_input})
                st.rerun()

            if st.session_state.chat_history:
                if st.button("Clear conversation", key="clear_chat"):
                    st.session_state.chat_history = []; st.rerun()

        with chat_layout_r:
            st.markdown(f'<div style="padding:12px;background:{T["surface2"]};border:1px solid {T["border"]};">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["ink3"]};letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;">Suggested Questions</div>', unsafe_allow_html=True)
            suggestions = [
                "Why is my stream score this value?",
                "What should I do before release?",
                "How do I pitch to Spotify playlists?",
                "Is my hook strong enough to go viral?",
                "How do I run Meta ads for music?",
                "Should I release now or wait?",
            ]
            for sugg in suggestions:
                if st.button(sugg, key=f"sugg_{sugg[:20]}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":sugg})
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════
elif page == "settings":
    st.markdown('<div class="page-pad">', unsafe_allow_html=True)
    st.markdown(f'''<div style="padding-top:28px;padding-bottom:20px;border-bottom:3px double {T["border2"]};margin-bottom:28px;">
      <div class="ed-kicker">Configuration</div>
      <div class="ed-title-sm">Settings</div>
      <div style="font-size:12px;color:{T["ink2"]};font-style:italic;margin-top:4px;">API keys and artist profile — saved for this session</div>
    </div>''', unsafe_allow_html=True)

    s1, s2 = st.columns([1.2, 1], gap="large")

    with s1:
        st.markdown(f'<div class="sec-head">API Keys <span>Required for analysis</span></div>', unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["ink3"]};letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;">OpenAI</div>', unsafe_allow_html=True)
            new_oai = st.text_input("OpenAI", value=st.session_state.openai_key, type="password",
                placeholder="sk-...", label_visibility="collapsed")
            if new_oai != st.session_state.openai_key:
                st.session_state.openai_key = new_oai
        with k2:
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["ink3"]};letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;">Claude</div>', unsafe_allow_html=True)
            new_claude = st.text_input("Claude", value=st.session_state.claude_key, type="password",
                placeholder="sk-ant-...", label_visibility="collapsed")
            if new_claude != st.session_state.claude_key:
                st.session_state.claude_key = new_claude
        with k3:
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:8px;color:{T["ink3"]};letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px;">Groq</div>', unsafe_allow_html=True)
            new_groq = st.text_input("Groq", value=st.session_state.groq_key, type="password",
                placeholder="gsk_...", label_visibility="collapsed")
            if new_groq != st.session_state.groq_key:
                st.session_state.groq_key = new_groq

        key_status = " · ".join(filter(None,[
            "✓ OpenAI" if st.session_state.openai_key else "",
            "✓ Claude" if st.session_state.claude_key else "",
            "✓ Groq"   if st.session_state.groq_key   else "",
        ])) or "⚠ No keys saved"
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};margin-top:8px;letter-spacing:.06em;">{key_status}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sec-head" style="margin-top:24px;">Artist Profile <span>Feeds into all analyses</span></div>', unsafe_allow_html=True)
        with st.form("artist_profile_form"):
            pf1, pf2 = st.columns(2)
            with pf1:
                p_ig  = st.number_input("Instagram Followers",0,10000000,int(st.session_state.instagram_val),100)
                p_sp  = st.number_input("Spotify Monthly Listeners",0,10000000,int(st.session_state.spotify_val),100)
            with pf2:
                p_yt  = st.number_input("YouTube Subscribers",0,10000000,int(st.session_state.youtube_val),100)
                p_bud = st.number_input("Marketing Budget (USD)",100,500000,int(st.session_state.budget_val),100)
            p_cs = st.selectbox("Career Stage",["emerging","growing","established"],
                index=["emerging","growing","established"].index(st.session_state.career_val))
            saved = st.form_submit_button("Save Profile", type="primary", use_container_width=True)
            if saved:
                st.session_state.instagram_val = p_ig
                st.session_state.spotify_val   = p_sp
                st.session_state.youtube_val   = p_yt
                st.session_state.budget_val    = p_bud
                st.session_state.career_val    = p_cs
                st.session_state.profile_saved = True
                st.rerun()
        if st.session_state.profile_saved:
            st.success("Profile saved.")

    with s2:
        st.markdown(f'<div class="sec-head">Saved Profile</div>', unsafe_allow_html=True)
        for ic,lb,vl in [
            ("📸","Instagram",  f"{st.session_state.instagram_val:,}"),
            ("🎧","Spotify",    f"{st.session_state.spotify_val:,}"),
            ("📺","YouTube",    f"{st.session_state.youtube_val:,}"),
            ("🎯","Stage",      st.session_state.career_val.title()),
            ("💵","Budget",     f"${st.session_state.budget_val:,}"),
        ]:
            st.markdown(f'''<div class="ed-card-sm" style="display:flex;align-items:center;gap:10px;">
              <span style="font-size:14px;">{ic}</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:10px;color:{T["ink3"]};flex:1;letter-spacing:.06em;">{lb}</span>
              <span style="font-family:IBM Plex Mono,monospace;font-size:11px;font-weight:500;color:{T["ink"]};">{vl}</span>
            </div>''', unsafe_allow_html=True)

        st.markdown(f'<div class="sec-head" style="margin-top:24px;">Team <span>StreamBreaker AI</span></div>', unsafe_allow_html=True)
        for badge,name,role in [("H","Harsh","M1 · XGBoost Prediction"),
                                  ("S","Stephanie","M2 · NLP Lyric Analysis"),
                                  ("M","Miguel","M3 · LLM Strategy"),
                                  ("GK","Gopi K.","M4 · Orchestration & UI")]:
            st.markdown(f'''<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid {T["border"]};">
              <div style="width:28px;height:28px;background:{T["ink"]};color:{T["surface"]};
                   display:flex;align-items:center;justify-content:center;
                   font-family:IBM Plex Mono,monospace;font-size:9px;font-weight:500;flex-shrink:0;">{badge}</div>
              <div>
                <div style="font-size:12px;font-weight:500;color:{T["ink"]};">{name}</div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{T["ink3"]};letter-spacing:.06em;">{role}</div>
              </div>
            </div>''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
