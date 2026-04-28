"""
StreamBreaker AI — v13
Original clean design + Model 0 audio upload + AI Chat
"""

import streamlit as st
import sys, os, json, urllib.request, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import StreamBreakerPipeline
from model0_audio import extract_features as extract_audio_features, get_file_metadata, KEY_NAMES

st.set_page_config(
    page_title="StreamBreaker AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SESSION STATE ─────────────────────────────────────────────
for k, v in [("result", None), ("chat_history", []),
             ("openai_key", ""), ("claude_key", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS (original design preserved) ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle { color: #6b7280; font-size: 1.1rem; margin-top: -8px; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 1px solid #e0e5ff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4f46e5; }
    .metric-label { font-size: 0.85rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        border-bottom: 2px solid #667eea;
        padding-bottom: 8px;
        margin-top: 24px;
    }
    .success-banner {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; padding: 12px 20px; border-radius: 10px;
        font-weight: 600; text-align: center;
    }
    .warning-banner {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white; padding: 12px 20px; border-radius: 10px;
        font-weight: 600; text-align: center;
    }
    .auto-banner {
        background: linear-gradient(135deg, rgba(16,185,129,.1), rgba(5,150,105,.07));
        border: 1px solid rgba(16,185,129,.3);
        border-radius: 8px; padding: 8px 12px;
        font-size: 0.8rem; color: #059669;
        margin-bottom: 8px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #eef0ff 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.6rem 1.2rem; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border-radius: 14px 14px 4px 14px;
        padding: 10px 14px; margin: 4px 0; margin-left: 20%;
        font-size: 0.9rem; line-height: 1.5;
    }
    .chat-ai {
        background: #f0f2ff; border: 1px solid #e0e5ff;
        border-radius: 14px 14px 14px 4px;
        padding: 10px 14px; margin: 4px 0; margin-right: 20%;
        font-size: 0.9rem; line-height: 1.5; color: #1f2937;
    }
    .chat-label {
        font-size: 0.7rem; color: #9ca3af; letter-spacing: 0.05em;
        text-transform: uppercase; margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(backend="openai", model=None, api_key=None):
    return StreamBreakerPipeline(backend=backend, model=model, api_key=api_key)

def fetch_lyrics_gpt(title, artist, api_key):
    if not api_key or not title: return "", "not_found"
    try:
        query = f"{artist} - {title}" if artist else title
        prompt = (f"Provide the full lyrics for: {query}\n\n"
                  "Format with [Verse 1], [Chorus], [Bridge] headers.\n"
                  "If unknown, reply only: UNKNOWN\nReturn only lyrics or UNKNOWN.")
        payload = json.dumps({"model":"gpt-3.5-turbo","max_tokens":1500,
            "messages":[{"role":"user","content":prompt}]}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
            data=payload, headers={"Content-Type":"application/json",
                                   "Authorization":f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"].strip()
        if "UNKNOWN" in text[:30] or len(text) < 30: return "", "not_found"
        return text, "gpt"
    except: return "", "not_found"

def do_chat(messages, system_prompt):
    """Try Claude → OpenAI → Groq in order."""
    claude_key = st.session_state.claude_key
    openai_key = st.session_state.openai_key
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        try:
            if not openai_key: openai_key = st.secrets.get("OPENAI_API_KEY", "")
        except: pass

    if claude_key:
        payload = json.dumps({"model":"claude-sonnet-4-20250514","max_tokens":800,
            "system":system_prompt,"messages":messages}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=payload, headers={"Content-Type":"application/json",
                                   "x-api-key":claude_key,"anthropic-version":"2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["content"][0]["text"].strip()
    elif openai_key:
        full = [{"role":"system","content":system_prompt}] + messages
        payload = json.dumps({"model":"gpt-3.5-turbo","max_tokens":800,"messages":full}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
            data=payload, headers={"Content-Type":"application/json",
                                   "Authorization":f"Bearer {openai_key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    raise Exception("No API key available — add one in the sidebar")


# ── HEADER ────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🎵 StreamBreaker AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Music Marketing Strategy Platform — Capstone Project</p>', unsafe_allow_html=True)

with st.expander("👥 Team Members", expanded=False):
    cols = st.columns(4)
    with cols[0]: st.markdown("**Model 0** — Miguel  \n*Audio Feature Extraction*")
    with cols[1]: st.markdown("**Model 1** — Harsh  \n*XGBoost Prediction*")
    with cols[2]: st.markdown("**Model 2** — Stephanie  \n*NLP Lyric Analysis*")
    with cols[3]: st.markdown("**Model 3** — Gopi Krishna  \n*LLM Marketing Strategy*")

st.divider()


# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Track Configuration")

# LLM Backend
st.sidebar.markdown("### 🤖 LLM Backend")
llm_backend = st.sidebar.selectbox("Backend",
    ["OpenAI", "Groq (Fast & Free)", "Ollama (Local)"], index=0)
backend_map = {"OpenAI":"openai","Groq (Fast & Free)":"groq","Ollama (Local)":"ollama"}
selected_backend = backend_map[llm_backend]

api_key_input = ""
if selected_backend != "ollama":
    api_key_input = os.getenv("OPENAI_API_KEY", "")
    try:
        if not api_key_input: api_key_input = st.secrets.get("OPENAI_API_KEY","")
    except: pass
    if not api_key_input: api_key_input = st.session_state.openai_key
    if api_key_input:
        st.sidebar.caption("✅ API key loaded")
    else:
        st.sidebar.warning("⚠️ No API key. Add in Settings expander below.")

# ── API Keys (optional override) ──
with st.sidebar.expander("🔑 API Keys (optional override)", expanded=False):
    new_oai = st.text_input("OpenAI Key", value=st.session_state.openai_key,
        type="password", placeholder="sk-...")
    if new_oai != st.session_state.openai_key:
        st.session_state.openai_key = new_oai
    new_claude = st.text_input("Claude Key (for chat)", value=st.session_state.claude_key,
        type="password", placeholder="sk-ant-...")
    if new_claude != st.session_state.claude_key:
        st.session_state.claude_key = new_claude
    st.caption("Keys are session-only and never stored.")

# ── MODEL 0: Audio Upload ──
st.sidebar.markdown("### 🎵 Model 0: Audio Upload")
st.sidebar.caption("Upload a track to auto-fill all audio features below")
audio_file = st.sidebar.file_uploader("Upload audio file",
    type=["mp3","wav","m4a","flac","ogg"], label_visibility="collapsed")

auto_vals = {}
auto_lyrics = ""
lyrics_source = ""

if audio_file:
    fname = re.sub(r'[\[\(]?\d{5,}[\]\)]?', '',
        audio_file.name.rsplit('.',1)[0].replace('_',' ')).strip()
    st.sidebar.audio(audio_file,
        format=f"audio/{audio_file.name.rsplit('.',1)[-1].lower()}")
    audio_file.seek(0)

    with st.sidebar:
        with st.spinner("Analyzing track…"):
            meta = get_file_metadata(audio_file, filename=audio_file.name)
            extracted, err = extract_audio_features(audio_file, filename=audio_file.name)
            # Lyrics
            embedded = meta.get("embedded_lyrics","")
            if embedded and len(embedded.strip()) > 20:
                auto_lyrics, lyrics_source = embedded.strip(), "embedded"
            elif api_key_input and selected_backend == "openai":
                auto_lyrics, lyrics_source = fetch_lyrics_gpt(
                    meta["title"], meta["artist"], api_key_input)
            else:
                auto_lyrics, lyrics_source = "", "not_found"

    if err:
        st.sidebar.error(f"Audio error: {err}")
    else:
        auto_vals = extracted
        key_str = f"{extracted['_key_name']} {extracted['_mode_name']}"
        lyr_note = {"embedded":"✓ Lyrics from file","gpt":"✓ Lyrics via GPT",
                    "not_found":"✏️ Paste lyrics below"}.get(lyrics_source,"")
        st.sidebar.markdown(f'''<div class="auto-banner">
          ✅ <strong>{fname}</strong><br>
          {key_str} · {extracted["tempo"]:.0f} BPM · {lyr_note}
        </div>''', unsafe_allow_html=True)

override = st.sidebar.toggle("Override auto-detected values",
    value=not bool(auto_vals)) if auto_vals else True
disabled = bool(auto_vals) and not override

# ── Audio Features ──
st.sidebar.markdown("### 🎧 Audio Features")
st.sidebar.caption("Auto-filled from upload · or set manually")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    danceability    = st.slider("Danceability",    0.0,  1.0, float(auto_vals.get("danceability",    0.65)), 0.01, disabled=disabled)
    energy          = st.slider("Energy",          0.0,  1.0, float(auto_vals.get("energy",          0.75)), 0.01, disabled=disabled)
    valence         = st.slider("Valence",         0.0,  1.0, float(auto_vals.get("valence",         0.55)), 0.01, disabled=disabled)
    acousticness    = st.slider("Acousticness",    0.0,  1.0, float(auto_vals.get("acousticness",    0.15)), 0.01, disabled=disabled)
with col_b:
    speechiness      = st.slider("Speechiness",      0.0,  1.0, float(auto_vals.get("speechiness",      0.05)), 0.01, disabled=disabled)
    instrumentalness = st.slider("Instrumentalness", 0.0,  1.0, float(auto_vals.get("instrumentalness", 0.00)), 0.01, disabled=disabled)
    liveness         = st.slider("Liveness",         0.0,  1.0, float(auto_vals.get("liveness",         0.12)), 0.01, disabled=disabled)
    loudness         = st.slider("Loudness (dB)",   -60.0, 0.0, float(auto_vals.get("loudness",        -6.0)),  0.5,  disabled=disabled)

tempo        = st.sidebar.slider("Tempo (BPM)", 60, 220, int(auto_vals.get("tempo", 125)), 1, disabled=disabled)
duration_sec = st.sidebar.slider("Duration (sec)", 30, 600, int(auto_vals.get("duration_ms", 210000)/1000), 5)
key          = st.sidebar.selectbox("Key", list(range(12)), index=int(auto_vals.get("key",5)),
                format_func=lambda x: KEY_NAMES[x], disabled=disabled)
mode         = st.sidebar.selectbox("Mode", [0,1], index=int(auto_vals.get("mode",1)),
                format_func=lambda x: "Major" if x==1 else "Minor", disabled=disabled)
time_signature = st.sidebar.selectbox("Time Signature", [3,4,5], index=1)
explicit       = st.sidebar.checkbox("Explicit Content", value=False)
genre          = st.sidebar.selectbox("Genre",
    ["indie","indie-pop","indie-rock","indie-folk","folk","acoustic",
     "alternative","singer-songwriter","dream-pop","lo-fi","pop",
     "rock","electronic","hip-hop","r-n-b"], index=1)

# ── Artist Profile ──
st.sidebar.markdown("### 👤 Artist Profile")
instagram         = st.sidebar.number_input("Instagram Followers",       0, 1000000, 1200, 100)
spotify_listeners = st.sidebar.number_input("Spotify Monthly Listeners", 0, 500000,  350,  50)
youtube_subs      = st.sidebar.number_input("YouTube Subscribers",       0, 1000000, 800,  100)
career_stage      = st.sidebar.selectbox("Career Stage", ["emerging","growing","established"])

st.sidebar.markdown("### 💰 Marketing Budget")
budget = st.sidebar.number_input("Budget (USD)", 100, 50000, 1500, 100)

st.sidebar.markdown("### 📝 Song Lyrics")
lyr_caption = {"embedded":"✓ Loaded from file — edit if needed",
               "gpt":"✓ Fetched via GPT — edit if needed",
               "not_found":"Paste lyrics below (not found automatically)"}.get(lyrics_source, "Paste lyrics here")
st.sidebar.caption(lyr_caption)
lyrics = st.sidebar.text_area("Lyrics",
    value=auto_lyrics or """[Verse 1]
I've been walking through the city lights
Chasing shadows in the neon nights

[Chorus]
Take me higher, take me higher
Set my heart on fire, fire
We're burning brighter through the rain
Take me higher once again""",
    height=200, label_visibility="collapsed")

# ── RUN BUTTON ──
run_clicked = st.sidebar.button("🚀 Run StreamBreaker AI",
    type="primary", use_container_width=True)

if run_clicked:
    audio_features = {
        "danceability":danceability,"energy":energy,"key":key,"loudness":loudness,
        "mode":mode,"speechiness":speechiness,"acousticness":acousticness,
        "instrumentalness":instrumentalness,"liveness":liveness,"valence":valence,
        "tempo":float(tempo),"duration_ms":duration_sec*1000,
        "time_signature":time_signature,"explicit":explicit,"genre":genre,
    }
    artist_profile = {
        "instagram_followers":instagram,"spotify_listeners":spotify_listeners,
        "youtube_subscribers":youtube_subs,"genre":genre.replace("-"," ").title(),
    }
    pipeline = load_pipeline(backend=selected_backend, api_key=api_key_input or None)
    progress = st.progress(0, text="Initializing pipeline…")
    progress.progress(10, text="📊 Model 1: XGBoost Prediction…")
    progress.progress(40, text="🎤 Model 2: NLP Analysis…")
    progress.progress(70, text="🚀 Model 3: Generating strategy…")
    result = pipeline.run(audio_features=audio_features, lyrics=lyrics,
                          budget=budget, artist_profile=artist_profile,
                          career_stage=career_stage)
    progress.progress(100, text="✅ Done!"); progress.empty()
    st.session_state.result = result
    st.session_state.chat_history = []  # reset chat for new analysis
    st.rerun()


# ── MAIN CONTENT ──────────────────────────────────────────────
if st.session_state.result is None:
    st.info("👈 Configure your track in the sidebar and click **Run StreamBreaker AI**")
    st.markdown("### 🔄 How It Works")
    flow_cols = st.columns(4)
    desc = [
        ("🎵 Model 0","Audio Extractor","Upload an mp3/wav","librosa extracts BPM, key, energy, and all Spotify-compatible features automatically"),
        ("📊 Model 1","XGBoost Prediction","Analyzes audio features","Predicts streaming success probability based on Spotify data"),
        ("🎤 Model 2","NLP Lyric Analysis","Processes your lyrics","Measures sentiment, hook strength, vocabulary richness"),
        ("🚀 Model 3","LLM Strategy","Uses real predictions","Generates a data-driven 4-week marketing plan"),
    ]
    for col, (icon, title, sub, body) in zip(flow_cols, desc):
        with col:
            st.markdown(f"#### {icon}\n**{title}**\n*{sub}*\n\n{body}")

else:
    result     = st.session_state.result
    prediction = result["model1_prediction"]
    nlp        = result["model2_nlp"]
    strategy   = result["model3_strategy"]
    pred_pct   = prediction["prediction_probability"]

    # ── TOP METRICS ──
    st.markdown("---")
    m1,m2,m3,m4 = st.columns(4)

    with m1:
        color = "#10b981" if pred_pct>=70 else "#f59e0b" if pred_pct>=50 else "#ef4444"
        st.markdown(f'''<div class="metric-card">
          <div class="metric-value" style="color:{color}">{pred_pct}%</div>
          <div class="metric-label">Stream Prediction</div>
        </div>''', unsafe_allow_html=True)

    with m2:
        sent_emoji = {"positive":"😊","negative":"😔","neutral":"😐"}.get(nlp["sentiment"],"😐")
        st.markdown(f'''<div class="metric-card">
          <div class="metric-value">{sent_emoji}</div>
          <div class="metric-label">Sentiment: {nlp["sentiment"]}</div>
        </div>''', unsafe_allow_html=True)

    with m3:
        hook = nlp["hook_repetition"]
        hook_color = "#10b981" if hook>=0.7 else "#f59e0b" if hook>=0.4 else "#ef4444"
        st.markdown(f'''<div class="metric-card">
          <div class="metric-value" style="color:{hook_color}">{hook:.0%}</div>
          <div class="metric-label">Hook Score</div>
        </div>''', unsafe_allow_html=True)

    with m4:
        st.markdown(f'''<div class="metric-card">
          <div class="metric-value" style="color:#4f46e5">${budget:,}</div>
          <div class="metric-label">Budget</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("---")

    # ── PREDICTION + NLP ──
    col_left, col_right = st.columns([1,1])

    with col_left:
        st.markdown('<div class="section-header">📊 Model 1: Prediction Details</div>',
                    unsafe_allow_html=True)
        if pred_pct >= 70:
            st.markdown('<div class="success-banner">✅ HIGH POTENTIAL — Invest in marketing!</div>',
                        unsafe_allow_html=True)
        elif pred_pct >= 50:
            st.markdown('<div class="warning-banner">⚠️ MODERATE — Proceed with caution</div>',
                        unsafe_allow_html=True)
        else:
            st.error("🔴 LOW POTENTIAL — Consider improving the track first")

        st.markdown(f"""
| Metric | Value |
|--------|-------|
| **Prediction Probability** | {pred_pct}% |
| **Will Hit 1K Streams** | {'✅ Yes' if prediction['will_hit_1k_streams'] else '❌ No'} |
| **Confidence** | {prediction['confidence']} |
| **Genre** | {genre.replace('-',' ').title()} |
""")

    with col_right:
        st.markdown('<div class="section-header">🎤 Model 2: Lyric Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
| Feature | Value | Interpretation |
|---------|-------|----------------|
| **Sentiment** | {nlp['sentiment']} | {'Good for mainstream' if nlp['sentiment']=='positive' else 'Target niche' if nlp['sentiment']=='negative' else 'Versatile targeting'} |
| **Lexical Diversity** | {nlp['lexical_diversity']:.2f} | {'Rich vocabulary' if nlp['lexical_diversity']>=0.7 else 'Moderate' if nlp['lexical_diversity']>=0.4 else 'Simple/catchy'} |
| **Hook Repetition** | {nlp['hook_repetition']:.2f} | {'🔥 HIGH viral potential!' if nlp['hook_repetition']>=0.7 else 'Moderate' if nlp['hook_repetition']>=0.4 else 'Weak hook'} |
| **Semantic Coherence** | {nlp['semantic_coherence']:.2f} | {'Strong theme' if nlp['semantic_coherence']>=0.6 else 'Mixed themes'} |
| **Profanity** | {'⚠️ Yes' if nlp['profanity_detected'] else '✅ Clean'} | {'May limit playlists' if nlp['profanity_detected'] else 'Playlist-friendly'} |
""")

    # ── STRATEGY ──
    st.markdown('<div class="section-header">🚀 Model 3: Marketing Strategy</div>',
                unsafe_allow_html=True)
    if strategy and strategy.get("success"):
        st.markdown(strategy["strategy"])
        st.divider()
        st.caption(f"✅ {strategy['metadata']['model']} · "
                   f"Tokens: {strategy['metadata']['tokens_used']} · "
                   f"Cost: ${strategy['metadata']['cost_estimate']:.4f}")
    elif strategy:
        st.error(f"❌ Strategy failed: {strategy.get('error','Unknown')}")

    if result.get("errors"):
        with st.expander("⚠️ Pipeline Warnings", expanded=False):
            for err in result["errors"]: st.warning(err)

    # ── AI CHAT ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">💬 Ask the AI About Your Results</div>',
                unsafe_allow_html=True)
    st.caption("The AI has full context of your analysis — ask about scores, strategy, release timing, or anything else.")

    chat_col, sugg_col = st.columns([3, 1])

    with sugg_col:
        st.markdown("**Quick questions:**")
        suggestions = [
            "Why is my stream score this value?",
            "What should I do before release?",
            "How do I pitch to Spotify playlists?",
            "Is my hook strong enough to go viral?",
            "How do I set up Meta Ads for music?",
            "Should I release now or wait?",
        ]
        for sugg in suggestions:
            if st.button(sugg, key=f"sugg_{sugg[:20]}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":sugg})
                st.rerun()

        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []; st.rerun()

    with chat_col:
        # Build system prompt with full analysis context
        system_prompt = (
            "You are StreamBreaker AI's music marketing analyst. "
            "You have the artist's full analysis results:\n"
            f"Stream Score: {pred_pct:.2f}% ({prediction['confidence']} confidence) · "
            f"Will hit 1K: {prediction['will_hit_1k_streams']} · "
            f"Hook: {nlp['hook_repetition']:.0%} · Sentiment: {nlp['sentiment']} · "
            f"Lexical diversity: {nlp['lexical_diversity']:.2f} · "
            f"Coherence: {nlp['semantic_coherence']:.2f} · "
            f"Energy: {auto_vals.get('energy', 0.75):.2f} · "
            f"Danceability: {auto_vals.get('danceability', 0.65):.2f} · "
            f"Tempo: {tempo} BPM · Key: {KEY_NAMES[key]} {'Major' if mode==1 else 'Minor'} · "
            f"Budget: ${budget:,} · Career stage: {career_stage} · "
            f"Genre: {genre} · Spotify listeners: {spotify_listeners:,}\n\n"
            "Be direct, specific, and reference the actual numbers in your answers. "
            "Keep responses concise and actionable."
        )

        # Display chat history
        for msg in st.session_state.chat_history:
            role = msg["role"]; text = msg["content"]
            if role == "user":
                st.markdown(f'<div class="chat-label">You</div><div class="chat-user">{text}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label">StreamBreaker AI</div><div class="chat-ai">{text}</div>',
                            unsafe_allow_html=True)

        # Process pending AI response
        if (st.session_state.chat_history and
                st.session_state.chat_history[-1]["role"] == "user"):
            chat_key = (st.session_state.claude_key or
                        st.session_state.openai_key or
                        os.getenv("OPENAI_API_KEY",""))
            try:
                if not chat_key: chat_key = st.secrets.get("OPENAI_API_KEY","")
            except: pass
            if not chat_key:
                st.warning("Add an API key in the sidebar to use the chat.")
            else:
                with st.spinner("Thinking…"):
                    try:
                        msgs = [{"role":m["role"],"content":m["content"]}
                                for m in st.session_state.chat_history]
                        reply = do_chat(msgs, system_prompt)
                        st.session_state.chat_history.append(
                            {"role":"assistant","content":reply})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Chat error: {str(e)[:120]}")

        # Input box
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        user_input = st.chat_input("Ask about your track, scores, or strategy…")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            st.rerun()
