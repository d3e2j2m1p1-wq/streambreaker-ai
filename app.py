"""
StreamBreaker AI — v8
Clean sidebar nav, audio player, all charts working.
Run: streamlit run app.py
"""
import streamlit as st
import sys, os, numpy as np, math, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import StreamBreakerPipeline
from model0_audio import extract_features as extract_audio_features, get_file_metadata, KEY_NAMES

st.set_page_config(page_title="StreamBreaker AI", page_icon="\U0001f3b5",
                   layout="wide", initial_sidebar_state="collapsed")

for k,v in [("result",None),("af",None),("auto_detected",False),
            ("auto_lyrics",""),("lyrics_source",""),("dark_mode",True),("page","analyze"),
            ("profile_saved",False),("instagram_val",1200),("spotify_val",350),
            ("youtube_val",800),("career_val","emerging"),("budget_val",1500),
            ("openai_key",""),("claude_key",""),("groq_key",""),
            ("chat_history",[]),("chat_llm","claude")]:
    if k not in st.session_state: st.session_state[k] = v

dark = st.session_state.dark_mode
page = st.session_state.page

if dark:
    T = {"app_bg":"#080b14","nav_bg":"#0d1121","nav_border":"#1a2035","nav_icon":"#2d3660",
         "card_bg":"#0f1525","card_border":"#1a2240","card_sm":"#0f1525","bar_bg":"#1a2240",
         "tp":"#f0f2ff","ts":"#3d4a6b","tb":"#8a93b8","tl":"#2d3660","tv":"#8090c0","tf":"#4a5580",
         "ib":"#0f1525","ibr":"#1a2240","ic":"#c0c8e8","st":"#0f1320","sth":"#2a3050",
         "tab_c":"#2d3660","tab_b":"#1a2240","sc":"#8a93b8","sh":"#b8c0e0","ss":"#c8d0f0",
         "tog":"\u2600\ufe0f Light Mode","plot_bg":"#0f1525","plot_grid":"#1a2240","plot_text":"#6a78a0",
         "upbg":"linear-gradient(135deg,rgba(108,99,255,.07),rgba(168,85,247,.05))"}
else:
    T = {"app_bg":"#f4f5fa","nav_bg":"#ffffff","nav_border":"#e2e5f0","nav_icon":"#b0b8d8",
         "card_bg":"#ffffff","card_border":"#e8ebf5","card_sm":"#f8f9fe","bar_bg":"#e8ebf5",
         "tp":"#0f1525","ts":"#8090b8","tb":"#4a5580","tl":"#a0a8cc","tv":"#6070a0","tf":"#7080a8",
         "ib":"#ffffff","ibr":"#dde0f0","ic":"#1a2240","st":"#eef0f8","sth":"#c8cce0",
         "tab_c":"#9098c0","tab_b":"#e2e5f0","sc":"#4a5580","sh":"#1a2240","ss":"#0f1525",
         "tog":"\U0001f319 Dark Mode","plot_bg":"#ffffff","plot_grid":"#e8ebf5","plot_text":"#8090b8",
         "upbg":"rgba(108,99,255,.03)"}

# ── SVG CHARTS ──────────────────────────────────────────────

def gauge_svg(pct, dark):
    """Semi-circle gauge for streaming probability."""
    c = "#10b981" if pct>=70 else "#f59e0b" if pct>=50 else "#ef4444"
    bg = T["bar_bg"]; txt = T["tp"]; sub = T["tl"]
    r=70; cx=100; cy=90; sw=12
    circ = 2*math.pi*r
    arc = circ*0.75
    dash_val = arc*(pct/100)
    dash_gap = arc - dash_val
    rot=-225
    label = "HIGH" if pct>=70 else "MODERATE" if pct>=50 else "LOW"
    return f"""<svg viewBox="0 0 200 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:220px;display:block;margin:0 auto;">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{bg}" stroke-width="{sw}"
    stroke-dasharray="{arc:.1f} {circ:.1f}" stroke-dashoffset="0"
    stroke-linecap="round" transform="rotate({rot} {cx} {cy})"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{c}" stroke-width="{sw}"
    stroke-dasharray="{dash_val:.1f} {circ:.1f}" stroke-dashoffset="0"
    stroke-linecap="round" transform="rotate({rot} {cx} {cy})"
    style="transition:stroke-dasharray .6s ease;filter:drop-shadow(0 0 4px {c}88)"/>
  <text x="{cx}" y="{cy-4}" text-anchor="middle" font-family="Inter"
    font-size="28" font-weight="800" fill="{c}">{pct}%</text>
  <text x="{cx}" y="{cy+16}" text-anchor="middle" font-family="Inter"
    font-size="10" fill="{sub}" letter-spacing="2">{label} POTENTIAL</text>
</svg>"""

def radar_svg(af, dark):
    """Octagon radar chart for 8 audio features."""
    labels = ["Dance","Energy","Valence","Acoustic","Speech","Instru.","Liveness","Loudness"]
    vals = [
        af["danceability"], af["energy"], af["valence"], af["acousticness"],
        af["speechiness"]*3,  # scale up for visibility
        1-af["instrumentalness"], af["liveness"],
        (af["loudness"]+60)/60
    ]
    vals = [min(1.0, max(0.0, v)) for v in vals]
    n = len(vals); cx=110; cy=105; r=75
    grid_c = T["bar_bg"]; line_c = "#6c63ff"; txt_c = T["tf"]
    bg = T["card_bg"]

    def pt(i, radius):
        angle = math.pi/2 - 2*math.pi*i/n
        return cx + radius*math.cos(angle), cy - radius*math.sin(angle)

    svg = f'<svg viewBox="0 0 220 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:240px;display:block;margin:0 auto;">'

    # Grid rings
    for ring in [0.25,0.5,0.75,1.0]:
        pts = " ".join([f"{pt(i,r*ring)[0]:.1f},{pt(i,r*ring)[1]:.1f}" for i in range(n)])
        svg += f'<polygon points="{pts}" fill="none" stroke="{grid_c}" stroke-width="1"/>'

    # Spokes
    for i in range(n):
        x,y = pt(i,r)
        svg += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{grid_c}" stroke-width="1"/>'

    # Data polygon
    data_pts = " ".join([f"{pt(i,r*vals[i])[0]:.1f},{pt(i,r*vals[i])[1]:.1f}" for i in range(n)])
    fill_c = "rgba(108,99,255,0.15)" if dark else "rgba(108,99,255,0.12)"
    svg += f'<polygon points="{data_pts}" fill="{fill_c}" stroke="#6c63ff" stroke-width="2" stroke-linejoin="round"/>'

    # Dots
    for i in range(n):
        x,y = pt(i,r*vals[i])
        svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#6c63ff" stroke="{bg}" stroke-width="1.5"/>'

    # Labels
    for i,lbl in enumerate(labels):
        x,y = pt(i,r*1.22)
        svg += f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="middle" font-family="Inter" font-size="9" fill="{txt_c}">{lbl}</text>'

    svg += '</svg>'
    return svg

def donut_svg(budget, dark):
    """Budget allocation donut: 65% Spotify, 25% TikTok, 10% Reserve."""
    slices = [("Spotify","#6c63ff",0.65),("TikTok","#a855f7",0.25),("Reserve","#3b82f6",0.10)]
    cx=80;cy=70;r=50;sw=22
    circ=2*math.pi*r; sub=T["tl"]; txt=T["tp"]
    # Pre-format monetary values to avoid comma breaking HTML parser
    bud_str   = f"{budget:,}"
    parts = []
    parts.append(f'<svg viewBox="0 0 220 150" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">')
    offset=0
    for label,color,pct in slices:
        dash=circ*pct; gap=circ-dash
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-dasharray="{dash:.1f} {gap:.1f}" stroke-dashoffset="{-offset:.1f}" transform="rotate(-90 {cx} {cy})" stroke-linecap="butt" opacity="0.9"/>')
        offset += circ*pct
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r-sw//2-2}" fill="{T["card_bg"]}"/>')
    parts.append(f'<text x="{cx}" y="{cy-3}" text-anchor="middle" font-family="Inter" font-size="12" font-weight="700" fill="{txt}">${bud_str}</text>')
    parts.append(f'<text x="{cx}" y="{cy+11}" text-anchor="middle" font-family="Inter" font-size="8" fill="{sub}">BUDGET</text>')
    lx=148
    for i,(label,color,pct) in enumerate(slices):
        ly=36+i*30
        amt_str = f"{int(budget*pct):,}"
        pct_str = f"{int(pct*100)}%"
        parts.append(f'<rect x="{lx}" y="{ly}" width="8" height="8" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx+12}" y="{ly+8}" font-family="Inter" font-size="10" fill="{txt}">{label}</text>')
        parts.append(f'<text x="{lx+12}" y="{ly+20}" font-family="Inter" font-size="9" fill="{sub}">${amt_str} ({pct_str})</text>')
    parts.append('</svg>')
    return "".join(parts)

def projection_svg(pp, budget, dark):
    """4-week projected stream timeline bars."""
    cps=0.022; mult=8.3
    total_low  = int((budget/cps)*0.6)
    total_high = int((budget/cps)*1.1)
    conf = pp/100
    weeks = [
        ("Wk 1", int(total_low*conf*0.15), int(total_high*conf*0.15)),
        ("Wk 2", int(total_low*conf*0.25), int(total_high*conf*0.25)),
        ("Wk 3", int(total_low*conf*0.30), int(total_high*conf*0.30)),
        ("Wk 4", int(total_low*conf*0.30), int(total_high*conf*0.30)),
    ]
    maxv = max(h for _,_,h in weeks) or 1
    W=200;H=120;pad_l=28;pad_b=22;pad_t=10
    bw=28; gap=12
    chart_w=W-pad_l-10; chart_h=H-pad_b-pad_t
    sub=T["tl"]; txt=T["tp"]; grid=T["bar_bg"]
    svg = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:220px;display:block;margin:0 auto;">'
    # Y gridlines
    for frac in [0.25,0.5,0.75,1.0]:
        y = H-pad_b-int(chart_h*frac)
        svg += f'<line x1="{pad_l}" y1="{y}" x2="{W-5}" y2="{y}" stroke="{grid}" stroke-width="1"/>'
        lbl = f"{int(maxv*frac/1000)}k"
        svg += f'<text x="{pad_l-3}" y="{y+3}" text-anchor="end" font-family="Inter" font-size="7" fill="{sub}">{lbl}</text>'
    # Bars
    for i,(label,lo,hi) in enumerate(weeks):
        x = pad_l + i*(bw+gap)
        # Low bar (solid)
        bh_lo = int(chart_h*(lo/maxv))
        y_lo = H-pad_b-bh_lo
        svg += f'<rect x="{x}" y="{y_lo}" width="{bw}" height="{bh_lo}" rx="3" fill="#6c63ff" opacity="0.5"/>'
        # High bar overlay
        bh_hi = int(chart_h*(hi/maxv))
        y_hi = H-pad_b-bh_hi
        svg += f'<rect x="{x+4}" y="{y_hi}" width="{bw-8}" height="{bh_hi}" rx="2" fill="#6c63ff"/>'
        svg += f'<text x="{x+bw//2}" y="{H-pad_b+12}" text-anchor="middle" font-family="Inter" font-size="8" fill="{sub}">{label}</text>'
    svg += '</svg>'
    return svg



def platform_roi_svg(pp, budget, dark):
    """Platform ROI comparison: Spotify vs TikTok vs Playlist expected streams."""
    cps = 0.022
    platforms = [
        ("Spotify Ads",   "#6c63ff", int(budget * 0.40 / cps * (pp/100))),
        ("Playlist Pitch","#a855f7", int(budget * 0.25 / cps * (pp/100) * 0.6)),
        ("TikTok",        "#3b82f6", int(budget * 0.25 / cps * (pp/100) * 0.4)),
        ("Reserve",       "#10b981", int(budget * 0.10 / cps * (pp/100) * 0.2)),
    ]
    maxv = max(v for _,_,v in platforms) or 1
    W=300; H=140; pad_l=90; pad_r=60; bar_h=16; gap=10
    pad_t=10
    bg=T["bar_bg"]; sub=T["tl"]; txt=T["tp"]
    svg=f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">'
    for i,(label,color,val) in enumerate(platforms):
        y = pad_t + i*(bar_h+gap)
        bw = int((W-pad_l-pad_r) * val/maxv)
        svg += f'<text x="{pad_l-6}" y="{y+bar_h//2+4}" text-anchor="end" font-family="Inter" font-size="10" fill="{sub}">{label}</text>'
        svg += f'<rect x="{pad_l}" y="{y}" width="{W-pad_l-pad_r}" height="{bar_h}" rx="4" fill="{bg}"/>'
        svg += f'<rect x="{pad_l}" y="{y}" width="{max(4,bw)}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>'
        k_val = f"{val//1000}k" if val>=1000 else str(val)
        svg += f'<text x="{pad_l+max(4,bw)+5}" y="{y+bar_h//2+4}" font-family="Inter" font-size="10" fill="{color}" font-weight="600">{k_val}</text>'
    svg += '</svg>'
    return svg

def weekly_projection_svg(pp, budget, dark):
    """4-week cumulative stream projection with low/high range."""
    cps=0.022
    total_lo = int(budget/cps*(pp/100)*0.7)
    total_hi = int(budget/cps*(pp/100)*1.3)
    # Cumulative week splits: 15%, 35%, 65%, 100%
    splits = [0.15, 0.35, 0.65, 1.0]
    weeks  = ["Wk 1","Wk 2","Wk 3","Wk 4"]
    lo_pts = [int(total_lo*s) for s in splits]
    hi_pts = [int(total_hi*s) for s in splits]
    maxv   = hi_pts[-1] or 1
    W=300;H=140;pl=36;pb=24;pt=10;pr=10
    cw=W-pl-pr; ch=H-pb-pt
    sub=T["tl"]; grid=T["bar_bg"]; txt=T["tp"]
    svg=f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">'
    # Gridlines
    for frac in [0.25,0.5,0.75,1.0]:
        y=H-pb-int(ch*frac)
        svg+=f'<line x1="{pl}" y1="{y}" x2="{W-pr}" y2="{y}" stroke="{grid}" stroke-width="1"/>'
        lbl=f"{int(maxv*frac/1000)}k"
        svg+=f'<text x="{pl-4}" y="{y+3}" text-anchor="end" font-family="Inter" font-size="8" fill="{sub}">{lbl}</text>'
    # Area fill between lo and hi
    xs=[pl+int(cw*i/3) for i in range(4)]
    hi_coords=" ".join([f"{xs[i]},{H-pb-int(ch*hi_pts[i]/maxv)}" for i in range(4)])
    lo_coords=" ".join([f"{xs[i]},{H-pb-int(ch*lo_pts[i]/maxv)}" for i in range(3,-1,-1)])
    svg+=f'<polygon points="{hi_coords} {lo_coords}" fill="rgba(108,99,255,0.15)"/>'
    # Hi line
    hi_line=" ".join([f"{xs[i]},{H-pb-int(ch*hi_pts[i]/maxv)}" for i in range(4)])
    svg+=f'<polyline points="{hi_line}" fill="none" stroke="#6c63ff" stroke-width="2" stroke-linejoin="round"/>'
    # Lo line
    lo_line=" ".join([f"{xs[i]},{H-pb-int(ch*lo_pts[i]/maxv)}" for i in range(4)])
    svg+=f'<polyline points="{lo_line}" fill="none" stroke="#a855f7" stroke-width="1.5" stroke-dasharray="4,3" stroke-linejoin="round"/>'
    # Dots + labels
    for i in range(4):
        x=xs[i]; y_hi=H-pb-int(ch*hi_pts[i]/maxv)
        svg+=f'<circle cx="{x}" cy="{y_hi}" r="3" fill="#6c63ff"/>'
        svg+=f'<text x="{x}" y="{H-pb+13}" text-anchor="middle" font-family="Inter" font-size="8" fill="{sub}">{weeks[i]}</text>'
    # Legend
    svg+=f'<text x="{pl}" y="{pt+8}" font-family="Inter" font-size="8" fill="#6c63ff">— High</text>'
    svg+=f'<text x="{pl+38}" y="{pt+8}" font-family="Inter" font-size="8" fill="#a855f7">- - Low</text>'
    svg+='</svg>'
    return svg

def key_metrics_svg(pp, nlp, af, dark):
    """At-a-glance score card comparing key metrics against indie-pop benchmarks."""
    txt=T["tp"]; sub=T["tl"]; bg=T["bar_bg"]; card=T["card_bg"]
    metrics = [
        ("Stream Score",   pp/100,      0.55, f"{pp:.0f}%"),
        ("Hook",           nlp.get("hook_repetition",0), 0.65, f"{nlp.get('hook_repetition',0):.0%}"),
        ("Energy",         af["energy"], 0.70, f"{af['energy']:.0%}"),
        ("Danceability",   af["danceability"], 0.65, f"{af['danceability']:.0%}"),
        ("Valence",        af["valence"], 0.50, f"{af['valence']:.0%}"),
    ]
    W=300; row=26; pad=14
    H = pad + len(metrics)*row + pad
    svg=f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">'
    for i,(name,val,bench,label) in enumerate(metrics):
        y = pad + i*row
        bar_w = W - 130 - 50
        # Label
        svg+=f'<text x="0" y="{y+14}" font-family="Inter" font-size="10" fill="{sub}">{name}</text>'
        # Track bar
        svg+=f'<rect x="90" y="{y+4}" width="{bar_w}" height="10" rx="4" fill="{bg}"/>'
        fill_w = int(bar_w * min(1.0, val))
        color = "#10b981" if val >= bench else "#f59e0b" if val >= bench*0.75 else "#ef4444"
        svg+=f'<rect x="90" y="{y+4}" width="{fill_w}" height="10" rx="4" fill="{color}" opacity="0.85"/>'
        # Benchmark tick
        bench_x = 90 + int(bar_w * bench)
        svg+=f'<rect x="{bench_x-1}" y="{y+2}" width="2" height="14" fill="{txt}" opacity="0.3"/>'
        # Value
        svg+=f'<text x="{W-5}" y="{y+14}" text-anchor="end" font-family="Inter" font-size="10" font-weight="600" fill="{color}">{label}</text>'
    # Benchmark legend
    svg+=f'<text x="90" y="{H-3}" font-family="Inter" font-size="8" fill="{sub}">│ = indie-pop benchmark</text>'
    svg+='</svg>'
    return svg

def nlp_bar_chart(nlp, dk):
    tc  = "#4a5580" if dk else "#8090b8"
    bg  = "#1a2240" if dk else "#e8ebf5"
    metrics = [
        ("Sentiment",        {"positive":1.0,"neutral":0.5,"negative":0.1}.get(nlp.get("sentiment","neutral"),0.5), "#6c63ff"),
        ("Hook Repetition",  nlp.get("hook_repetition", 0),  "#a855f7"),
        ("Lexical Diversity",nlp.get("lexical_diversity", 0), "#3b82f6"),
        ("Sem. Coherence",   nlp.get("semantic_coherence",0), "#10b981"),
    ]
    W, row_h, bar_h, lbl_w = 310, 38, 12, 115
    H = row_h * len(metrics) + 16
    bars = ""
    for i, (name, val, col) in enumerate(metrics):
        y = 8 + i * row_h
        bw = max(0, min(1, val)) * (W - lbl_w - 20)
        bars += (
            f'<text x="{lbl_w-6}" y="{y+bar_h//2+4}" text-anchor="end" font-size="10" fill="{tc}" font-family="DM Sans">{name}</text>'
            f'<rect x="{lbl_w}" y="{y}" width="{W-lbl_w-20}" height="{bar_h}" rx="4" fill="{bg}"/>'
            f'<rect x="{lbl_w}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="4" fill="{col}" opacity="0.85"/>'
            f'<text x="{lbl_w+bw+5:.1f}" y="{y+bar_h//2+4}" font-size="10" fill="{col}" font-family="DM Sans" font-weight="600">{val:.0%}</text>'
        )
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

def feat_bar(label, value, fmt=".2f", max_val=1.0):
    pct = min(100,max(0,(value/max_val)*100)) if max_val else 0
    st.markdown(f'''<div class="feat-row"><span class="feat-label">{label}</span>
      <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct}%"></div></div>
      <span class="feat-val">{value:{fmt}}</span></div>''', unsafe_allow_html=True)

def fetch_lyrics_gpt(title, artist, api_key):
    if not api_key or not title: return "", "not_found"
    try:
        import urllib.request, json
        query = f"{artist} - {title}" if artist else title
        prompt = (f"Provide the full lyrics for: {query}\n\n"
                  "Format with [Verse 1], [Chorus], [Bridge] headers. "
                  "If unknown, reply UNKNOWN only.")
        payload = json.dumps({"model":"gpt-3.5-turbo","max_tokens":1500,
                              "messages":[{"role":"user","content":prompt}]}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
            data=payload, headers={"Content-Type":"application/json",
                                   "Authorization":f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"].strip()
        return ("", "not_found") if "UNKNOWN" in text[:30] or len(text)<30 else (text,"gpt")
    except: return "", "not_found"

@st.cache_resource
def load_pipeline(backend="openai", api_key=None):
    return StreamBreakerPipeline(backend=backend, api_key=api_key)

# ── CSS ──────────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
.stApp{{background:{T["app_bg"]};color:{T["tp"]};}}
[data-testid="stHeader"],[data-testid="stToolbar"],footer{{display:none;}}
.main .block-container{{padding:20px 24px 40px 24px!important;max-width:100%!important;}}
[data-testid="stMainBlockContainer"]{{padding:20px 24px!important;max-width:100%!important;}}
::-webkit-scrollbar{{width:4px;}}::-webkit-scrollbar-track{{background:{T["st"]};}}
::-webkit-scrollbar-thumb{{background:{T["sth"]};border-radius:2px;}}
/* Sidebar styling */

section[data-testid="stSidebar"] .block-container{{padding:20px 12px!important;}}
.sb-logo-wrap{{display:flex;align-items:center;gap:10px;margin-bottom:24px;padding:0 4px;}}
.sb-logo{{width:36px;height:36px;background:linear-gradient(135deg,#6c63ff,#a855f7);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;box-shadow:0 0 16px rgba(108,99,255,.35);flex-shrink:0;}}
.sb-logo-text{{font-family:var(--display-font);font-size:14px;font-weight:800;color:{T["tp"]};line-height:1.2;}}
.sb-logo-sub{{font-size:10px;color:{T["tl"]};}}

.stButton>button[kind="primary"]{{background:linear-gradient(135deg,#6c63ff,#a855f7)!important;border:none!important;border-radius:9px!important;font-family:var(--display-font)!important;font-weight:700!important;font-size:13px!important;color:white!important;box-shadow:0 4px 18px rgba(108,99,255,.3)!important;width:100%!important;}}
.sb-title{{font-family:var(--display-font);font-size:20px;font-weight:800;color:{T["tp"]};line-height:1.2;margin-bottom:3px;}}
.sb-subtitle{{font-size:12px;color:{T["ts"]};margin-bottom:16px;}}
.sb-label{{font-family:var(--display-font);font-size:9px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:{T["tl"]};margin-bottom:10px;margin-top:20px;}}
.sb-label:first-child{{margin-top:0;}}
.sb-card{{background:{T["card_bg"]};border:1px solid {T["card_border"]};border-radius:13px;padding:15px;margin-bottom:10px;}}
.sb-card-sm{{background:{T["card_sm"]};border:1px solid {T["card_border"]};border-radius:9px;padding:11px 13px;margin-bottom:7px;}}
.auto-banner{{background:{"rgba(16,185,129,.1)" if dark else "rgba(16,185,129,.07)"};border:1px solid rgba(16,185,129,.25);border-radius:9px;padding:9px 12px;font-size:11px;color:#16a34a;margin-bottom:12px;display:flex;align-items:center;gap:7px;flex-wrap:wrap;}}
.sb-pill{{display:inline-flex;align-items:center;gap:4px;background:rgba(108,99,255,.12);border:1px solid rgba(108,99,255,.25);border-radius:20px;padding:2px 9px;font-size:11px;font-weight:600;color:#6c63ff;}}
.sb-pill.green{{background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.25);color:#10b981;}}
.sb-pill.amber{{background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.25);color:#d97706;}}
.sb-pill.red{{background:rgba(239,68,68,.1);border-color:rgba(239,68,68,.25);color:#dc2626;}}
.feat-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;}}
.feat-label{{font-size:11px;color:{T["tf"]};min-width:108px;}}
.feat-bar-bg{{flex:1;height:3px;background:{T["bar_bg"]};border-radius:2px;margin:0 9px;}}
.feat-bar-fill{{height:3px;border-radius:2px;background:linear-gradient(90deg,#6c63ff,#a855f7);}}
.feat-val{{font-size:11px;color:{T["tv"]};font-weight:600;min-width:34px;text-align:right;}}
.nlp-badge{{display:flex;align-items:center;gap:9px;background:{T["card_bg"]};border:1px solid {T["card_border"]};border-radius:9px;padding:10px 13px;font-size:12px;color:{T["tb"]};margin-bottom:7px;}}
.model-badge{{display:inline-flex;align-items:center;gap:5px;background:{T["card_sm"]};border:1px solid {T["card_border"]};border-radius:5px;padding:2px 7px;font-size:10px;font-weight:700;color:{T["tf"]};letter-spacing:.06em;text-transform:uppercase;}}
.model-badge .dot{{width:5px;height:5px;border-radius:50%;background:#6c63ff;}}
.genre-tag{{display:inline-block;background:rgba(108,99,255,.12);border:1px solid rgba(108,99,255,.25);border-radius:5px;padding:2px 8px;font-size:10px;font-weight:700;color:#6c63ff;text-transform:uppercase;letter-spacing:.06em;}}
.waveform{{display:flex;align-items:center;gap:2px;height:28px;margin:5px 0;}}
.waveform .bar{{flex:1;background:linear-gradient(180deg,#6c63ff,#a855f7);border-radius:1px;opacity:.55;animation:wave 1.2s ease-in-out infinite alternate;}}
@keyframes wave{{from{{transform:scaleY(.3);}}to{{transform:scaleY(1);}}}}
.strategy-body{{font-size:13px;line-height:1.85;color:{T["sc"]};}}
.strategy-body h1,.strategy-body h2,.strategy-body h3{{font-family:var(--display-font)!important;font-size:13px!important;font-weight:700!important;color:{T["sh"]}!important;margin-top:14px!important;margin-bottom:5px!important;}}
.strategy-body strong{{color:{T["ss"]};}} .strategy-body ul{{padding-left:15px;}} .strategy-body li{{margin-bottom:3px;}}
[data-testid="stSelectbox"]>div>div{{background:{T["ib"]}!important;border:1px solid {T["ibr"]}!important;border-radius:7px!important;color:{T["ic"]}!important;font-size:13px!important;}}
[data-testid="stNumberInput"]>div>div>input{{background:{T["ib"]}!important;border:1px solid {T["ibr"]}!important;color:{T["ic"]}!important;border-radius:7px!important;font-size:12px!important;}}
textarea{{background:{T["ib"]}!important;border:1px solid {T["ibr"]}!important;color:{T["ic"]}!important;border-radius:9px!important;font-size:12px!important;font-family:'DM Sans',sans-serif!important;}}
[data-testid="stSlider"]>div>div>div>div{{background:#6c63ff!important;}}
[data-testid="stSlider"]>div>div>div{{background:{T["bar_bg"]}!important;}}
.stSlider label{{font-size:11px!important;color:{T["tf"]}!important;}}
[data-testid="stTabs"]>div:first-child{{border-bottom:1px solid {T["tab_b"]}!important;}}
[data-testid="stTabs"] button{{font-family:var(--display-font)!important;font-size:11px!important;font-weight:700!important;color:{T["tab_c"]}!important;letter-spacing:.05em!important;padding:7px 14px!important;border-radius:0!important;background:transparent!important;}}
[data-testid="stTabs"] button[aria-selected="true"]{{color:#6c63ff!important;border-bottom:2px solid #6c63ff!important;}}
[data-testid="stProgress"]>div>div{{background:linear-gradient(90deg,#6c63ff,#a855f7)!important;border-radius:3px!important;}}
[data-testid="stProgress"]>div{{background:{T["bar_bg"]}!important;border-radius:3px!important;height:3px!important;}}
[data-testid="stExpander"]{{background:{T["card_bg"]}!important;border:1px solid {T["card_border"]}!important;border-radius:9px!important;}}
[data-testid="stFileUploader"] section{{background:{T["upbg"]}!important;border:1.5px dashed rgba(108,99,255,.3)!important;border-radius:12px!important;}}
[data-testid="stCheckbox"] label,[data-testid="stToggle"] label{{font-size:12px!important;color:{T["tf"]}!important;}}
audio{{width:100%!important;height:36px!important;border-radius:8px!important;accent-color:#6c63ff;}}
</style>""", unsafe_allow_html=True)

# ── TOP NAV BAR ──────────────────────────────────────────────
nav_bg  = T["nav_bg"]
nav_bdr = T["nav_border"]
acc     = "#6c63ff"

# Logo + nav buttons + theme toggle all in one row
n0, n1, n2, n3, n4, n5, n6 = st.columns([1.2, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7])

with n0:
    st.markdown(f'''<div style="display:flex;align-items:center;gap:8px;padding:6px 0;">
      <div style="width:30px;height:30px;background:linear-gradient(135deg,#6c63ff,#a855f7);
           border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;">🎵</div>
      <span style="font-size:13px;font-weight:700;color:{T["tp"]};">StreamBreaker</span>
    </div>''', unsafe_allow_html=True)

with n1:
    active = page == "analyze"
    if st.button("🎵  Analyze",  key="nav_analyze",  use_container_width=True,
                 type="primary" if active else "secondary"):
        st.session_state.page = "analyze"; st.rerun()

with n2:
    active = page == "visuals"
    if st.button("📊  Visuals",  key="nav_visuals",  use_container_width=True,
                 type="primary" if active else "secondary"):
        st.session_state.page = "visuals"; st.rerun()

with n3:
    active = page == "chat"
    if st.button("💬  Chat", key="nav_chat", use_container_width=True,
                 type="primary" if active else "secondary"):
        st.session_state.page = "chat"; st.rerun()

with n4:
    active = page == "settings"
    if st.button("⚙️  Settings", key="nav_settings", use_container_width=True,
                 type="primary" if active else "secondary"):
        st.session_state.page = "settings"; st.rerun()

with n5:
    done_count = sum([
        st.session_state.auto_detected,
        st.session_state.result is not None,
        st.session_state.result is not None,
        st.session_state.result is not None,
    ])
    clr = "#10b981" if done_count == 4 else "#f59e0b" if done_count > 0 else T["tl"]
    st.markdown(f'''<div style="padding:6px 0;text-align:center;">
      <div style="font-size:10px;color:{T["tl"]};">Pipeline</div>
      <div style="font-size:13px;font-weight:700;color:{clr};">{done_count}/4 ✓</div>
    </div>''', unsafe_allow_html=True)

with n6:
    if st.button(T["tog"], key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Divider under nav
st.markdown(f'<div style="height:1px;background:{T["nav_border"]};margin:4px 0 16px 0;"></div>', unsafe_allow_html=True)

# ── PAGE ROUTING ─────────────────────────────────────────────
# ── PAGE ROUTING ─────────────────────────────────────────────
if page == "settings":
    st.markdown(f'<div class="sb-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-subtitle">API keys · artist profile · saved for this session</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── API Keys ──
    st.markdown(f'<div class="sb-label" style="margin-top:0;">API Keys</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f'<div style="font-size:11px;color:{T["tf"]};margin-bottom:4px;">🤖 Claude (Anthropic)</div>', unsafe_allow_html=True)
        new_claude = st.text_input("Claude API Key", value=st.session_state.claude_key,
            type="password", placeholder="sk-ant-...", label_visibility="collapsed")
        if new_claude != st.session_state.claude_key:
            st.session_state.claude_key = new_claude
    with k2:
        st.markdown(f'<div style="font-size:11px;color:{T["tf"]};margin-bottom:4px;">🟢 OpenAI</div>', unsafe_allow_html=True)
        new_oai = st.text_input("OpenAI API Key", value=st.session_state.openai_key,
            type="password", placeholder="sk-...", label_visibility="collapsed")
        if new_oai != st.session_state.openai_key:
            st.session_state.openai_key = new_oai
    with k3:
        st.markdown(f'<div style="font-size:11px;color:{T["tf"]};margin-bottom:4px;">⚡ Groq (free/fast)</div>', unsafe_allow_html=True)
        new_groq = st.text_input("Groq API Key", value=st.session_state.groq_key,
            type="password", placeholder="gsk_...", label_visibility="collapsed")
        if new_groq != st.session_state.groq_key:
            st.session_state.groq_key = new_groq

    # Show which keys are active
    key_status = []
    if st.session_state.claude_key: key_status.append("✅ Claude")
    if st.session_state.openai_key: key_status.append("✅ OpenAI")
    if st.session_state.groq_key:   key_status.append("✅ Groq")
    if not key_status:               key_status = ["⚠️ No keys saved — add at least one above"]
    st.markdown(f'<div style="font-size:11px;color:{T["tl"]};margin:8px 0 20px;">{" &nbsp;·&nbsp; ".join(key_status)}</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{T["nav_border"]};margin:4px 0 20px;"></div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns([1.3, 1])
    with sc1:
        st.markdown(f'<div class="sb-label" style="margin-top:0">Artist Profile</div>', unsafe_allow_html=True)
        with st.form("artist_profile_form"):
            fi1, fi2 = st.columns(2)
            with fi1:
                p_ig  = st.number_input("Instagram Followers",       0, 10000000, int(st.session_state.instagram_val), 100)
                p_sp  = st.number_input("Spotify Monthly Listeners", 0, 10000000, int(st.session_state.spotify_val),   100)
            with fi2:
                p_yt  = st.number_input("YouTube Subscribers",       0, 10000000, int(st.session_state.youtube_val),  100)
                p_bud = st.number_input("Marketing Budget (USD)",    100, 500000,  int(st.session_state.budget_val),   100)
            p_cs = st.selectbox("Career Stage", ["emerging","growing","established"],
                index=["emerging","growing","established"].index(st.session_state.career_val))
            saved = st.form_submit_button("💾  Save Profile", type="primary", use_container_width=True)
            if saved:
                st.session_state.instagram_val = p_ig
                st.session_state.spotify_val   = p_sp
                st.session_state.youtube_val   = p_yt
                st.session_state.budget_val    = p_bud
                st.session_state.career_val    = p_cs
                st.session_state.profile_saved = True
                st.rerun()
        if st.session_state.profile_saved:
            st.success("✅ Profile saved — used on next analysis run.")

    with sc2:
        st.markdown(f'<div class="sb-label" style="margin-top:0">Saved Profile</div>', unsafe_allow_html=True)
        for ic,lb,vl in [
            ("📸","Instagram",       f"{st.session_state.instagram_val:,}"),
            ("🎧","Spotify Monthly", f"{st.session_state.spotify_val:,}"),
            ("📺","YouTube",         f"{st.session_state.youtube_val:,}"),
            ("🎯","Career Stage",    st.session_state.career_val.title()),
            ("💵","Budget",          f"${st.session_state.budget_val:,}"),
        ]:
            st.markdown(f'<div class="sb-card-sm" style="display:flex;align-items:center;gap:9px;">'
                f'<span style="font-size:14px;">{ic}</span>'
                f'<span style="font-size:12px;color:{T["tf"]};flex:1;">{lb}</span>'
                f'<span style="font-size:12px;font-weight:600;color:{T["tv"]};">{vl}</span></div>',
                unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{T["nav_border"]};margin:28px 0 20px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-label">Team</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    for i,(badge,name,role,desc) in enumerate([
        ("H","Harsh","Model 1 · XGBoost","Trained an XGBoost classifier on Spotify audio features to predict 1K-stream threshold performance within 90 days."),
        ("S","Stephanie","Model 2 · NLP","TextBlob pipeline extracting sentiment, hook repetition, lexical diversity, and semantic coherence."),
        ("M","Miguel","Model 3 · LLM Strategy","GPT-3.5 prompt pipeline synthesizing all model outputs into actionable marketing plans."),
        ("GK","Gopi K.","Model 4 · Orchestration","Unified pipeline and original Streamlit interface."),
    ]):
        with [t1,t2][i % 2]:
            st.markdown(f'''<div class="sb-card" style="display:flex;gap:12px;align-items:flex-start;margin-bottom:8px;">
              <div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#6c63ff,#a855f7);
                   display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:white;flex-shrink:0;">{badge}</div>
              <div><div style="font-size:13px;font-weight:600;color:{T["tp"]};margin-bottom:2px;">{name}</div>
                <div style="font-size:10px;color:#6c63ff;margin-bottom:4px;">{role}</div>
                <div style="font-size:11px;color:{T["tb"]};line-height:1.55;">{desc}</div></div>
            </div>''', unsafe_allow_html=True)

elif page == "chat":
    st.markdown(f'<div class="sb-title">💬 AI Assistant</div>', unsafe_allow_html=True)

    has_result = st.session_state.result is not None
    has_audio  = st.session_state.af is not None

    # Build system prompt with full analysis context
    def build_system_prompt():
        base = """You are StreamBreaker AI's music marketing assistant. You have deep knowledge of music marketing, Spotify algorithms, TikTok virality, playlist pitching, and the indie-pop industry.

You have access to the artist's full analysis results and you answer questions about their specific track, scores, strategy, and what actions to take next. Be direct, specific, and use the data you have. Keep responses concise but actionable — this is a busy artist, not a research paper."""

        if has_result:
            result = st.session_state.result
            af     = st.session_state.af
            pred   = result["model1_prediction"]
            nlp    = result["model2_nlp"]
            strat  = result["model3_strategy"]
            ctx = f"""

CURRENT TRACK ANALYSIS:
- Stream Score: {pred["prediction_probability"]}% ({pred["confidence"]} confidence)
- Will hit 1K streams: {"Yes" if pred["will_hit_1k_streams"] else "No"}
- Key: {KEY_NAMES[af["key"]]} {"Major" if af["mode"]==1 else "Minor"} | Tempo: {af["tempo"]:.0f} BPM
- Energy: {af["energy"]:.2f} | Danceability: {af["danceability"]:.2f} | Valence: {af["valence"]:.2f}
- Acousticness: {af["acousticness"]:.2f} | Speechiness: {af["speechiness"]:.2f} | Liveness: {af["liveness"]:.2f}
- Loudness: {af["loudness"]:.1f} dB | Instrumentalness: {af["instrumentalness"]:.2f}
- Sentiment: {nlp["sentiment"]} | Hook Repetition: {nlp["hook_repetition"]:.2f} | Lexical Diversity: {nlp["lexical_diversity"]:.2f}
- Semantic Coherence: {nlp["semantic_coherence"]:.2f} | Profanity: {"Yes" if nlp["profanity_detected"] else "No"}
- Artist: {st.session_state.career_val.title()} stage | Instagram: {st.session_state.instagram_val:,} | Spotify Monthly: {st.session_state.spotify_val:,} | Budget: ${st.session_state.budget_val:,}
- Genre: selected by artist"""
            return base + ctx
        else:
            return base + "\n\nNo analysis has been run yet. You can still answer general music marketing questions."

    # Determine which key to use for chat
    chat_key = st.session_state.claude_key or st.session_state.openai_key or st.session_state.groq_key
    chat_provider = "claude" if st.session_state.claude_key else "openai" if st.session_state.openai_key else "groq" if st.session_state.groq_key else None

    # Chat layout
    ch_main, ch_side = st.columns([3, 1])

    with ch_side:
        st.markdown(f'<div class="sb-card" style="padding:14px;">', unsafe_allow_html=True)
        st.markdown(f'<div class="sb-label" style="margin-top:0;">Context</div>', unsafe_allow_html=True)
        if has_result:
            result = st.session_state.result
            pred   = result["model1_prediction"]
            pp     = pred["prediction_probability"]
            clr    = "#10b981" if pp>=70 else "#f59e0b" if pp>=50 else "#ef4444"
            st.markdown(f'''<div style="text-align:center;padding:10px 0;">
              <div style="font-size:28px;font-weight:700;color:{clr};">{pp:.0f}%</div>
              <div style="font-size:10px;color:{T["tl"]};">STREAM SCORE</div>
            </div>''', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:11px;color:{T["tb"]};line-height:1.7;">'
                f'🎵 Analysis loaded<br>'
                f'🤖 Using {chat_provider or "no key"}<br>'
                f'💬 {len(st.session_state.chat_history)//2} messages</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:11px;color:{T["tl"]};">Run an analysis first for track-specific answers.<br><br>General music marketing questions work without it.</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear chat", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

        # Suggested questions
        st.markdown(f'<div class="sb-label">Try asking</div>', unsafe_allow_html=True)
        suggestions = []
        if has_result:
            pp = st.session_state.result["model1_prediction"]["prediction_probability"]
            suggestions = [
                "Why is my stream score this high/low?",
                "How do I improve my hook score?",
                "Which platform should I prioritize?",
                "Is my track ready to release?",
                "How do I pitch to Spotify playlists?",
                "What should my TikTok strategy be?",
            ]
        else:
            suggestions = [
                "How does Spotify's algorithm work?",
                "What makes a song go viral on TikTok?",
                "How do I pitch to playlist curators?",
                "What's a good marketing budget for indie-pop?",
            ]
        for q in suggestions:
            if st.button(q, key=f"sugg_{q[:20]}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with ch_main:
        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(f'''<div style="text-align:center;padding:40px 20px;color:{T["tl"]};">
                  <div style="font-size:36px;margin-bottom:12px;">💬</div>
                  <div style="font-size:14px;font-weight:600;color:{T["tf"]};margin-bottom:6px;">
                    Ask me anything about your track
                  </div>
                  <div style="font-size:12px;">
                    {"I have your full analysis loaded — ask about scores, strategy, next steps." if has_result else "Run an analysis first, or ask general music marketing questions."}
                  </div>
                </div>''', unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    role = msg["role"]
                    text = msg["content"]
                    if role == "user":
                        st.markdown(f'''<div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
                          <div style="background:linear-gradient(135deg,#6c63ff,#a855f7);color:white;
                               border-radius:14px 14px 4px 14px;padding:10px 14px;max-width:75%;
                               font-size:13px;line-height:1.6;">{text}</div>
                        </div>''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''<div style="display:flex;justify-content:flex-start;margin-bottom:10px;">
                          <div style="background:{T["card_bg"]};border:1px solid {T["card_border"]};
                               border-radius:14px 14px 14px 4px;padding:10px 14px;max-width:80%;
                               font-size:13px;line-height:1.7;color:{T["tb"]};">{text}</div>
                        </div>''', unsafe_allow_html=True)

        # Process pending message (from suggestion buttons or input)
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            if not chat_key:
                st.warning("Add an API key in ⚙️ Settings to enable the chat.")
            else:
                with st.spinner("Thinking…"):
                    import urllib.request, json as _json
                    sys_prompt = build_system_prompt()
                    msgs = [{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_history]
                    try:
                        if chat_provider == "claude":
                            payload = _json.dumps({
                                "model": "claude-sonnet-4-20250514",
                                "max_tokens": 800,
                                "system": sys_prompt,
                                "messages": msgs
                            }).encode()
                            req = urllib.request.Request(
                                "https://api.anthropic.com/v1/messages",
                                data=payload,
                                headers={
                                    "Content-Type":"application/json",
                                    "x-api-key": chat_key,
                                    "anthropic-version":"2023-06-01"
                                }
                            )
                            with urllib.request.urlopen(req, timeout=30) as r:
                                data = _json.loads(r.read())
                            reply = data["content"][0]["text"].strip()
                        else:
                            # OpenAI / Groq compatible
                            base_url = "https://api.groq.com/openai/v1/chat/completions" if chat_provider=="groq" else "https://api.openai.com/v1/chat/completions"
                            model    = "llama3-8b-8192" if chat_provider=="groq" else "gpt-3.5-turbo"
                            full_msgs = [{"role":"system","content":sys_prompt}] + msgs
                            payload = _json.dumps({"model":model,"max_tokens":800,"messages":full_msgs}).encode()
                            req = urllib.request.Request(base_url, data=payload,
                                headers={"Content-Type":"application/json","Authorization":f"Bearer {chat_key}"})
                            with urllib.request.urlopen(req, timeout=30) as r:
                                data = _json.loads(r.read())
                            reply = data["choices"][0]["message"]["content"].strip()

                        st.session_state.chat_history.append({"role":"assistant","content":reply})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Chat error: {str(e)[:120]}")

        # Input box always at bottom
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        user_input = st.chat_input("Ask about your track, scores, strategy, next steps…")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            st.rerun()

elif page == "visuals":
    st.markdown(f'<div class="sb-title">📊 Visual Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-subtitle">Detailed charts from the last analysis run</div>', unsafe_allow_html=True)
    if st.session_state.result is None:
        st.info("Run an analysis on the Analyze page first to see charts here.")
    else:
        result = st.session_state.result
        af     = st.session_state.af
        pred   = result["model1_prediction"]
        nlp    = result["model2_nlp"]
        pp     = pred["prediction_probability"]
        bud    = st.session_state.get("budget_val", 1500)

        # ── Row 1: Gauge · Radar · Donut ──
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown(f'<div class="sb-card" style="text-align:center;padding:18px 12px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;text-align:center;">Stream Score</div>', unsafe_allow_html=True)
            st.markdown(gauge_svg(pp, dark), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with v2:
            st.markdown(f'<div class="sb-card" style="padding:14px 10px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;text-align:center;">Audio Radar</div>', unsafe_allow_html=True)
            st.markdown(radar_svg(af, dark), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with v3:
            st.markdown(f'<div class="sb-card" style="padding:14px 10px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;text-align:center;">Budget Allocation</div>', unsafe_allow_html=True)
            st.markdown(donut_svg(bud, dark), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Row 2: Projection · Platform ROI ──
        r2a, r2b = st.columns(2)
        with r2a:
            st.markdown(f'<div class="sb-card" style="padding:14px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;">4-Week Stream Projection</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:10px;color:{T["tl"]};margin-bottom:8px;">Estimated cumulative streams — solid line = optimistic, dashed = conservative</div>', unsafe_allow_html=True)
            st.markdown(weekly_projection_svg(pp, bud, dark), unsafe_allow_html=True)
            lo = int(bud/0.022*(pp/100)*0.7); hi = int(bud/0.022*(pp/100)*1.3)
            st.markdown(f'<div style="font-size:11px;color:{T["tv"]};text-align:center;margin-top:6px;font-weight:600;">{lo:,} – {hi:,} projected streams</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with r2b:
            st.markdown(f'<div class="sb-card" style="padding:14px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;">Expected Streams by Platform</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:10px;color:{T["tl"]};margin-bottom:8px;">Based on $0.022 indie-pop CPS benchmark and {pp:.0f}% stream probability</div>', unsafe_allow_html=True)
            st.markdown(platform_roi_svg(pp, bud, dark), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Row 3: Key metrics vs benchmark · NLP · Audio bars ──
        r3a, r3b, r3c = st.columns(3)
        with r3a:
            st.markdown(f'<div class="sb-card" style="padding:14px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;">vs. Indie-Pop Benchmark</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:10px;color:{T["tl"]};margin-bottom:8px;">Bar shows your track · tick = genre average</div>', unsafe_allow_html=True)
            st.markdown(key_metrics_svg(pp, nlp, af, dark), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with r3b:
            st.markdown(f'<div class="sb-card" style="padding:14px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;">NLP Score Breakdown</div>', unsafe_allow_html=True)
            st.markdown(nlp_bar_chart(nlp, dark), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with r3c:
            st.markdown(f'<div class="sb-card" style="padding:14px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-label" style="margin-top:0;">Audio Features</div>', unsafe_allow_html=True)
            feat_bar("Danceability",     af["danceability"])
            feat_bar("Energy",           af["energy"])
            feat_bar("Valence",          af["valence"])
            feat_bar("Acousticness",     af["acousticness"])
            feat_bar("Speechiness",      af["speechiness"])
            feat_bar("Instrumentalness", af["instrumentalness"])
            feat_bar("Liveness",         af["liveness"])
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # ── PROFILE SETUP POPUP (first run only) ──
    if not st.session_state.profile_saved:
        with st.expander("👤 Set up your artist profile before running analysis", expanded=True):
            st.markdown(f'<div style="font-size:12px;color:{T["tb"]};margin-bottom:12px;">These values are used by all models. You can update them anytime in ⚙️ Settings.</div>', unsafe_allow_html=True)
            pp1, pp2, pp3 = st.columns(3)
            with pp1:
                p_ig = st.number_input("Instagram Followers", 0, 10000000, st.session_state.instagram_val, 100, key="popup_ig")
                p_sp = st.number_input("Spotify Monthly Listeners", 0, 10000000, st.session_state.spotify_val, 100, key="popup_sp")
            with pp2:
                p_yt = st.number_input("YouTube Subscribers", 0, 10000000, st.session_state.youtube_val, 100, key="popup_yt")
                p_bud = st.number_input("Marketing Budget (USD)", 100, 500000, st.session_state.budget_val, 100, key="popup_bud")
            with pp3:
                p_cs = st.selectbox("Career Stage", ["emerging","growing","established"], key="popup_cs")
                st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
                if st.button("Save & Continue →", type="primary", use_container_width=True, key="popup_save"):
                    st.session_state.instagram_val = p_ig
                    st.session_state.spotify_val   = p_sp
                    st.session_state.youtube_val   = p_yt
                    st.session_state.budget_val    = p_bud
                    st.session_state.career_val    = p_cs
                    st.session_state.profile_saved = True
                    st.rerun()

    # ── ANALYZE PAGE ─────────────────────────────────────────
    col_l, col_c, col_r = st.columns([320, 680, 260], gap="small")
    with col_l:
        st.markdown("<div style='padding:28px 16px;'>", unsafe_allow_html=True)
        st.markdown('<div class="sb-title">StreamBreaker AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-subtitle">Music Marketing Intelligence</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-label">LLM Engine</div>', unsafe_allow_html=True)
        llm_choice = st.selectbox("Backend",["Claude (Anthropic)","OpenAI","Groq (Fast & Free)","Ollama (Local)"],label_visibility="collapsed")
        backend_map = {"Claude (Anthropic)":"claude","OpenAI":"openai","Groq (Fast & Free)":"groq","Ollama (Local)":"ollama"}
        selected_backend = backend_map[llm_choice]
        api_key_input = ""
        if selected_backend != "ollama":
            # Check settings keys first, then fall back to env/secrets
            if selected_backend == "openai":
                api_key_input = st.session_state.openai_key or os.getenv("OPENAI_API_KEY","")
                try:
                    if not api_key_input: api_key_input = st.secrets["OPENAI_API_KEY"]
                except: pass
            elif selected_backend == "groq":
                api_key_input = st.session_state.groq_key or os.getenv("GROQ_API_KEY","")
            elif selected_backend == "claude":
                api_key_input = st.session_state.claude_key or os.getenv("ANTHROPIC_API_KEY","")
            if not api_key_input:
                st.warning(f"⚠️ No {selected_backend} key — add one in ⚙️ Settings.")

        st.markdown('<div class="sb-label">Track Upload (Model 0)</div>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio",type=["mp3","wav","m4a","flac","ogg"],label_visibility="collapsed")

        # ── Audio player ──
        if audio_file:
            import re as _re
            fname = _re.sub(r'[\[\(]?\d{5,}[\]\)]?','', audio_file.name.rsplit('.',1)[0].replace('_',' ')).strip()
            st.markdown(f'<div class="sb-label" style="margin-top:10px;">Now Playing</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sb-card" style="padding:10px 12px;">', unsafe_allow_html=True)
            st.markdown(f'''<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
              <div style="width:26px;height:26px;border-radius:7px;background:linear-gradient(135deg,#6c63ff,#a855f7);
                   display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;">🎵</div>
              <div style="font-size:12px;font-weight:600;color:{T["tp"]};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{fname}</div>
            </div>''', unsafe_allow_html=True)
            audio_file.seek(0)
            st.audio(audio_file, format=f"audio/{audio_file.name.rsplit('.',1)[-1].lower()}")
            st.markdown('</div>', unsafe_allow_html=True)
            audio_file.seek(0)

        auto_vals = {}
        if audio_file:
            with st.spinner("🎙 Analyzing full track — this may take 20–40 seconds…"):
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
                lyr_badge={"embedded":"💾 in file","gpt":"🤖 via GPT","not_found":"✏️ paste below"}.get(lyr_source,"")
                st.markdown(f'''<div class="auto-banner">✅ <strong>{extracted["_key_name"]} {extracted["_mode_name"]}</strong>
                  &nbsp;·&nbsp; {extracted["tempo"]:.0f} BPM
                  {"&nbsp;·&nbsp; "+lyr_badge if lyr_badge else ""}
                </div>''', unsafe_allow_html=True)
                if meta["title"]:
                    st.markdown(f'<div style="font-size:11px;color:{T["tf"]};margin-bottom:8px;">🎵 <strong style="color:{T["tv"]};">{meta["title"]}</strong>{"  ·  "+meta["artist"] if meta["artist"] else ""}</div>', unsafe_allow_html=True)
        else:
            st.session_state.auto_detected = False
            st.session_state.auto_lyrics = ""
            st.session_state.lyrics_source = ""

        disabled = bool(auto_vals) and not st.toggle("Override with manual values", value=not bool(auto_vals))

        st.markdown('<div class="sb-label">Audio Features</div>', unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            danceability=st.slider("Danceability",0.0,1.0,float(auto_vals.get("danceability",0.65)),0.01,disabled=disabled)
            energy=st.slider("Energy",0.0,1.0,float(auto_vals.get("energy",0.75)),0.01,disabled=disabled)
            valence=st.slider("Valence",0.0,1.0,float(auto_vals.get("valence",0.55)),0.01,disabled=disabled)
            acousticness=st.slider("Acousticness",0.0,1.0,float(auto_vals.get("acousticness",0.15)),0.01,disabled=disabled)
        with c2:
            speechiness=st.slider("Speechiness",0.0,1.0,float(auto_vals.get("speechiness",0.05)),0.01,disabled=disabled)
            instrumentalness=st.slider("Instrumental.",0.0,1.0,float(auto_vals.get("instrumentalness",0.0)),0.01,disabled=disabled)
            liveness=st.slider("Liveness",0.0,1.0,float(auto_vals.get("liveness",0.12)),0.01,disabled=disabled)
            loudness=st.slider("Loudness (dB)",-60.0,0.0,float(auto_vals.get("loudness",-6.0)),0.5,disabled=disabled)
        tempo=st.slider("Tempo (BPM)",60,220,int(auto_vals.get("tempo",125)),1,disabled=disabled)
        duration_sec=st.slider("Duration (sec)",30,600,int(auto_vals.get("duration_ms",210000)/1000),5)
        c3,c4=st.columns(2)
        with c3:
            key=st.selectbox("Key",list(range(12)),index=int(auto_vals.get("key",5)),format_func=lambda x:KEY_NAMES[x],disabled=disabled)
            time_sig=st.selectbox("Time Sig",[3,4,5],index=1)
        with c4:
            mode=st.selectbox("Mode",[0,1],index=int(auto_vals.get("mode",1)),format_func=lambda x:"Major" if x==1 else "Minor",disabled=disabled)
            explicit=st.checkbox("Explicit",value=False)
        genre=st.selectbox("Genre",["indie","indie-pop","indie-rock","indie-folk","folk","acoustic",
            "alternative","singer-songwriter","dream-pop","lo-fi","pop","rock","electronic","hip-hop","r-n-b"],index=1)

        # Artist profile pulled from Settings — no inputs here
        career_stage     = st.session_state.career_val
        instagram        = st.session_state.instagram_val
        spotify_listeners= st.session_state.spotify_val
        youtube_subs     = st.session_state.youtube_val
        budget           = st.session_state.budget_val
        st.markdown(f'''<div style="display:flex;align-items:center;justify-content:space-between;
          background:{"rgba(108,99,255,.08)" if dark else "rgba(108,99,255,.05)"};
          border:1px solid rgba(108,99,255,.2);border-radius:9px;padding:9px 12px;margin-bottom:12px;">
          <div style="font-size:11px;color:{T["tv"]};font-weight:600;">
            🎯 {career_stage.title()} &nbsp;·&nbsp; 💰 ${budget:,}
          </div>
          <div style="font-size:10px;color:{T["tl"]};">
            Edit in ⚙️ Settings
          </div>
        </div>''', unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Song Lyrics</div>', unsafe_allow_html=True)
        ls=st.session_state.lyrics_source
        if ls=="gpt": st.markdown(f'<div style="font-size:11px;color:{T["tf"]};margin-bottom:4px;">🤖 Lyrics fetched via GPT — edit if needed:</div>', unsafe_allow_html=True)
        elif ls=="embedded": st.markdown(f'<div style="font-size:11px;color:{T["tf"]};margin-bottom:4px;">💾 Lyrics from file — edit if needed:</div>', unsafe_allow_html=True)
        elif ls=="not_found": st.markdown(f'<div style="font-size:11px;color:{T["tf"]};margin-bottom:4px;">✏️ Not found automatically — paste below:</div>', unsafe_allow_html=True)
        lyrics=st.text_area("Lyrics",height=150,label_visibility="collapsed",value=st.session_state.auto_lyrics,
            placeholder="[Verse 1]\nPaste lyrics here...\n\n[Chorus]\nSection headers improve NLP analysis.")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        run_btn=st.button("⚡  Run Analysis",type="primary",use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # RUN PIPELINE
    # RUN
    if run_btn:
        af_input={"danceability":danceability,"energy":energy,"key":key,"loudness":loudness,
                  "mode":mode,"speechiness":speechiness,"acousticness":acousticness,
                  "instrumentalness":instrumentalness,"liveness":liveness,"valence":valence,
                  "tempo":float(tempo),"duration_ms":duration_sec*1000,"time_signature":time_sig,
                  "explicit":explicit,"genre":genre}
        artist_profile={"instagram_followers":instagram,"spotify_listeners":spotify_listeners,
                        "youtube_subscribers":youtube_subs,"genre":genre.replace("-"," ").title()}
        st.session_state["budget_val"] = budget
        pipeline=load_pipeline(backend=selected_backend,api_key=api_key_input or None)
        with col_c:
            prog=st.progress(0,text="Initializing…")
            prog.progress(15,text="📊 Model 1 · XGBoost…")
        try:
            prog.progress(45,text="🎤 Model 2 · NLP…")
            prog.progress(72,text="🚀 Model 3 · Strategy…")
            result=pipeline.run(audio_features=af_input,lyrics=lyrics,budget=budget,
                                artist_profile=artist_profile,career_stage=career_stage)
            prog.progress(100,text="✅ Done")
            prog.empty()
            st.session_state.result=result
            st.session_state.af=af_input
        except Exception as e:
            prog.empty()
            with col_c: st.error(f"Pipeline error: {e}")

    # CENTER PANEL

    # CENTER + RIGHT
    # CENTER PANEL
    with col_c:
        st.markdown("<div style='padding-top:28px;'>", unsafe_allow_html=True)
        if st.session_state.result is None:
            st.markdown(f"""<div style="display:flex;flex-direction:column;align-items:center;
              justify-content:center;min-height:75vh;text-align:center;padding:0 50px;">
              <div style="font-size:56px;margin-bottom:18px;opacity:.12;">🎵</div>
              <div style="font-family:Plus Jakarta Sans,sans-serif;font-size:18px;font-weight:800;color:{T["tl"]};margin-bottom:7px;">Ready to Analyze</div>
              <div style="font-size:13px;color:{T["tl"]};max-width:280px;line-height:1.7;">
                Configure your track on the left, hit <strong style="color:{T["ts"]};">Run Analysis</strong>,
                then check the <strong style="color:{T["ts"]};">📊 Visuals</strong> page for charts.
              </div>
              <div style="margin-top:24px;display:flex;gap:16px;flex-wrap:wrap;justify-content:center;">
                {"".join([f'<div style="text-align:center;"><div style="font-family:Plus Jakarta Sans,sans-serif;font-size:16px;font-weight:800;color:{T["tl"]};">{m}</div><div style="font-size:10px;color:{T["tl"]};margin-top:1px;">{n}</div></div><div style="color:{T["tl"]};font-size:16px;align-self:center;">→</div>' for m,n in [("M0","Audio"),("M1","XGBoost"),("M2","NLP"),("M3","Strategy")]])[:-len(f'<div style="color:{T["tl"]};font-size:16px;align-self:center;">→</div>')]}
              </div></div>""", unsafe_allow_html=True)
        else:
            result=st.session_state.result; af=st.session_state.af
            pred=result["model1_prediction"]; nlp=result["model2_nlp"]; strat=result["model3_strategy"]
            pp=pred["prediction_probability"]
            pred_c="#10b981" if pp>=70 else "#d97706" if pp>=50 else "#dc2626"
            hook=nlp["hook_repetition"]; hook_c="#10b981" if hook>=0.7 else "#d97706" if hook>=0.4 else "#dc2626"
            sent_e={"positive":"😊","negative":"😔","neutral":"😐"}.get(nlp["sentiment"],"😐")

            k1,k2,k3,k4=st.columns(4)
            for ck,ic,vl,lb,cl in [(k1,"📊",f"{pp}%","Stream Score",pred_c),(k2,"🎣",f"{hook:.0%}","Hook Score",hook_c),(k3,sent_e,nlp["sentiment"].title(),"Sentiment","#6c63ff"),(k4,"💰",f"${budget:,}","Budget",T["tv"])]:
                with ck:
                    st.markdown(f'<div class="sb-card" style="text-align:center;padding:13px 8px;"><div style="font-size:18px;margin-bottom:3px;">{ic}</div><div style="font-family:var(--display-font);font-size:20px;font-weight:800;color:{cl};line-height:1;">{vl}</div><div style="font-size:9px;color:{T["tl"]};text-transform:uppercase;letter-spacing:.08em;margin-top:3px;">{lb}</div></div>', unsafe_allow_html=True)

            # Visuals nudge
            st.markdown(f'<div style="font-size:11px;color:{T["tl"]};margin:6px 0 10px;text-align:right;">📊 See full charts on the <strong style="color:#6c63ff;">Visuals</strong> page →</div>', unsafe_allow_html=True)

            tab_pred,tab_nlp,tab_strat=st.tabs(["📊  Prediction","🎤  Lyric Analysis","🚀  Strategy"])

            with tab_pred:
                pa,pb=st.columns([1,1])
                with pa:
                    st.markdown(f'<div class="sb-card" style="padding:16px 12px;">', unsafe_allow_html=True)
                    st.markdown(gauge_svg(pp,dark), unsafe_allow_html=True)
                    if pp>=70: st.success("🔥 **HIGH POTENTIAL** — Invest in marketing")
                    elif pp>=50: st.warning("⚠️ **MODERATE** — Targeted spend recommended")
                    else: st.error("🔴 **LOW POTENTIAL** — Refine track before release")
                    st.markdown(f'<div style="text-align:center;margin-top:6px;"><span class="sb-pill {'green' if pp>=70 else 'amber' if pp>=50 else 'red'}">{pred["confidence"]} confidence</span></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with pb:
                    st.markdown(f'<div class="sb-card" style="padding:14px 12px;">', unsafe_allow_html=True)
                    st.markdown(f'<div class="sb-label" style="margin-top:0;margin-bottom:8px;">Audio Radar</div>', unsafe_allow_html=True)
                    st.markdown(radar_svg(af,dark), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                d1,d2,d3,d4=st.columns(4)
                for cd,lb,vl in [(d1,"Tempo",f"{af['tempo']:.0f} BPM"),(d2,"Key",f"{KEY_NAMES[af['key']]} {'Maj' if af['mode']==1 else 'Min'}"),(d3,"Duration",f"{af['duration_ms']//1000//60}:{(af['duration_ms']//1000)%60:02d}"),(d4,"Genre",genre.replace('-',' ').title())]:
                    with cd: st.markdown(f'<div class="sb-card-sm" style="text-align:center;"><div style="font-size:9px;color:{T["tl"]};text-transform:uppercase;">{lb}</div><div style="font-family:var(--display-font);font-size:13px;font-weight:700;color:{T["tp"]};margin-top:2px;">{vl}</div></div>', unsafe_allow_html=True)

            with tab_nlp:
                na,nb=st.columns([1,1])
                with na:
                    st.markdown(f'<div class="sb-label" style="margin-top:0">Lyric Metrics</div>', unsafe_allow_html=True)
                    for ic,lb,vl,it in [("🎭","Sentiment",nlp["sentiment"].title(),""),("📖","Lexical Diversity",f"{nlp['lexical_diversity']:.2f}","Rich" if nlp['lexical_diversity']>=0.7 else "Moderate" if nlp['lexical_diversity']>=0.4 else "Simple"),("🔁","Hook Repetition",f"{nlp['hook_repetition']:.2f}","🔥 High viral" if nlp['hook_repetition']>=0.7 else "Moderate" if nlp['hook_repetition']>=0.4 else "Weak"),("🧵","Semantic Coherence",f"{nlp['semantic_coherence']:.2f}","Strong theme" if nlp['semantic_coherence']>=0.6 else "Mixed"),("🚫","Profanity","Detected" if nlp['profanity_detected'] else "Clean","")]:
                        st.markdown(f'<div class="nlp-badge"><span style="font-size:16px;">{ic}</span><div><div style="font-size:10px;color:{T["tl"]};">{lb}</div><div style="font-weight:600;color:{T["tp"]};">{vl}</div></div><span style="font-size:11px;color:{T["tl"]};margin-left:auto;">{it}</span></div>', unsafe_allow_html=True)
                with nb:
                    st.markdown(f'<div class="sb-label" style="margin-top:0">Insights</div>', unsafe_allow_html=True)
                    sent=nlp["sentiment"]; lex=nlp["lexical_diversity"]; hk=nlp["hook_repetition"]
                    ins=[]
                    if sent=="positive": ins.append("**Positive sentiment** broadens mainstream appeal and playlist eligibility.")
                    elif sent=="negative": ins.append("**Negative sentiment** connects with niche emotional communities.")
                    else: ins.append("**Neutral sentiment** gives versatile targeting across formats.")
                    if lex>=0.7: ins.append("**Rich vocabulary** signals artistic depth — ideal for press pitching.")
                    elif lex<0.4: ins.append("**Simple vocabulary** maximizes sing-along — lean into short-form video.")
                    if hk>=0.7: ins.append("**Strong hook** — prioritize TikTok and Reels clips.")
                    elif hk<0.4: ins.append("**Weak hook** — rewrite chorus for more repetition.")
                    if nlp["profanity_detected"]: ins.append("**Explicit content** may limit editorial placement.")
                    for i in ins: st.markdown(f'<div class="sb-card-sm" style="font-size:12px;color:{T["tb"]};line-height:1.65;">{i}</div>', unsafe_allow_html=True)

            with tab_strat:
                if strat and strat.get("success"):
                    ms=strat["metadata"]
                    st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;"><span class="model-badge"><span class="dot"></span>Model 3 · LLM</span><span class="sb-pill">{ms["model"]}</span><span style="font-size:10px;color:{T["tl"]};">{ms["tokens_used"]} tokens · ${ms["cost_estimate"]:.4f}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="strategy-body">{strat["strategy"]}</div>', unsafe_allow_html=True)
                elif strat: st.error(f"Strategy failed: {strat.get('error','Unknown')}")
            if result.get("errors"):
                with st.expander("⚠️ Pipeline Warnings"):
                    for e in result["errors"]: st.warning(e)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT PANEL
    with col_r:
        st.markdown("<div style='padding-top:28px;'>", unsafe_allow_html=True)
        st.markdown(f'<div class="sb-label" style="margin-top:0">Track Snapshot</div>', unsafe_allow_html=True)
        if st.session_state.af:
            af=st.session_state.af
            dur=f"{af['duration_ms']//1000//60}:{(af['duration_ms']//1000)%60:02d}"
            bars="".join([f'<div class="bar" style="height:{h}px;animation-delay:{d}s"></div>' for h,d in [(11,0),(19,.1),(15,.2),(25,.05),(13,.15),(21,.25),(17,.1),(23,.3)]])
            atag=f'<div class="auto-banner" style="margin-top:9px;margin-bottom:0;font-size:10px;">🎙 Auto-extracted</div>' if st.session_state.auto_detected else ""
            st.markdown(f'<div class="sb-card"><div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;"><span class="genre-tag">{genre}</span><span style="font-size:10px;color:{T["tl"]};">{dur}</span></div><div class="waveform">{bars}</div><div style="display:flex;gap:14px;margin-top:11px;"><div><div style="font-size:9px;color:{T["tl"]};">BPM</div><div style="font-family:var(--display-font);font-size:17px;font-weight:800;color:{T["tp"]};">{af["tempo"]:.0f}</div></div><div><div style="font-size:9px;color:{T["tl"]};">KEY</div><div style="font-family:var(--display-font);font-size:17px;font-weight:800;color:{T["tp"]};">{KEY_NAMES[af["key"]]}{"M" if af["mode"]==1 else "m"}</div></div><div><div style="font-size:9px;color:{T["tl"]};">ENERGY</div><div style="font-family:var(--display-font);font-size:17px;font-weight:800;color:{T["tp"]};">{af["energy"]:.0%}</div></div></div>{atag}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sb-card" style="text-align:center;padding:22px 14px;"><div style="font-size:28px;opacity:.12;margin-bottom:7px;">🎵</div><div style="font-size:11px;color:{T["tl"]};">Run analysis to see snapshot</div></div>', unsafe_allow_html=True)

        # Artist profile is in Settings (⚙️)
        st.markdown(f'<div class="sb-label">Artist</div>', unsafe_allow_html=True)
        for ic,lb,vl in [
            ("📸","Instagram", f"{st.session_state.instagram_val:,}"),
            ("🎧","Spotify",   f"{st.session_state.spotify_val:,}"),
            ("📺","YouTube",   f"{st.session_state.youtube_val:,}"),
            ("🎯","Stage",     st.session_state.career_val.title()),
            ("💵","Budget",    f"${st.session_state.budget_val:,}"),
        ]:
            st.markdown(f'<div class="sb-card-sm" style="display:flex;align-items:center;gap:9px;">'
                f'<span style="font-size:13px;">{ic}</span>'
                f'<span style="font-size:11px;color:{T["tf"]};flex:1;">{lb}</span>'
                f'<span style="font-size:12px;font-weight:600;color:{T["tv"]};">{vl}</span></div>',
                unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:10px;color:{T["tl"]};margin-top:4px;text-align:center;">Edit in ⚙️ Settings</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
