import streamlit as st
import requests
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="News Analysis System",
    page_icon="📰",
    layout="wide"
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .entity-tag {
        display: inline-block;
        background: #e8f4fd;
        color: #1f77b4;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .sentiment-positive { color: #28a745; font-weight: 700; font-size: 1.2rem; }
    .sentiment-negative { color: #dc3545; font-weight: 700; font-size: 1.2rem; }
    .sentiment-neutral  { color: #6c757d; font-weight: 700; font-size: 1.2rem; }
    .category-badge {
        background: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 6px 18px;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📰 News Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered text classification and information extraction</div>', unsafe_allow_html=True)

# ── API Health Check ──────────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    device = health.get("device", "unknown")
    st.success(f"✅ API connected | Running on: **{device.upper()}**")
except Exception:
    st.error("❌ API not reachable. Make sure FastAPI is running on port 8000.")
    st.code("python -m uvicorn api.app:app --reload --port 8000")
    st.stop()

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown("---")
input_mode = st.radio(
    "Choose input method:",
    ["📝 Enter Text", "🔗 Enter URL"],
    horizontal=True
)

text_input = ""
url_input  = ""

if input_mode == "📝 Enter Text":
    text_input = st.text_area(
        "Paste your news article text below:",
        height=180,
        placeholder="e.g. Apple reported record revenue of $25 billion in Q1 2024..."
    )
else:
    url_input = st.text_input(
        "Enter a news article URL:",
        placeholder="e.g. https://www.bbc.com/news/articles/..."
    )

analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if input_mode == "📝 Enter Text" and len(text_input.strip()) < 20:
        st.warning("⚠️ Please enter at least 20 characters.")
    elif input_mode == "🔗 Enter URL" and not url_input.strip():
        st.warning("⚠️ Please enter a URL.")
    else:
        with st.spinner("🤖 Analyzing..."):
            try:
                if input_mode == "📝 Enter Text":
                    response = requests.post(
                        f"{API_URL}/analyze",
                        json={"text": text_input},
                        timeout=30
                    )
                else:
                    response = requests.post(
                        f"{API_URL}/analyze-url",
                        json={"url": url_input},
                        timeout=30
                    )

                if response.status_code != 200:
                    st.error(f"❌ API Error: {response.json().get('detail', 'Unknown error')}")
                    st.stop()

                data = response.json()

                # Handle URL response structure
                if "title" in data:
                    st.info(f"📰 **{data['title']}**")

                classification = data["classification"]
                extraction     = data["extraction"]
                sentiment      = extraction["sentiment"]
                entities       = extraction["entities"]
                keywords       = extraction["keywords"]

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try a shorter text.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        st.markdown("---")

        # ── Row 1: Classification + Sentiment ─────────────────────────────────
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown('<div class="section-header">📂 Category</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-card">'
                f'<span class="category-badge">{classification["category"]}</span>'
                f'<br><br><b>Confidence:</b> {classification["confidence"]}%'
                f'</div>',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown('<div class="section-header">💬 Sentiment</div>', unsafe_allow_html=True)
            label = sentiment["label"]
            emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "⚪"
            css   = f"sentiment-{label}"
            st.markdown(
                f'<div class="metric-card">'
                f'{emoji} <span class="{css}">{label.upper()}</span>'
                f'<br><br><b>Confidence:</b> {sentiment["confidence"]}%'
                f'</div>',
                unsafe_allow_html=True
            )

        with col3:
            st.markdown('<div class="section-header">📊 Category Scores</div>', unsafe_allow_html=True)
            scores  = classification["all_scores"]
            fig = go.Figure(go.Bar(
                x=list(scores.values()),
                y=list(scores.keys()),
                orientation="h",
                marker_color="#1f77b4"
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=150,
                xaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Row 2: Entities + Keywords ─────────────────────────────────────────
        col4, col5 = st.columns([1, 1])

        with col4:
            st.markdown('<div class="section-header">🏷️ Named Entities</div>', unsafe_allow_html=True)
            if entities:
                ENTITY_COLORS = {
                    "PERSON":  "#d4edda",
                    "ORG":     "#cce5ff",
                    "GPE":     "#fff3cd",
                    "LOC":     "#fff3cd",
                    "MONEY":   "#d4edda",
                    "PERCENT": "#f8d7da",
                    "DATE":    "#e2d9f3",
                    "PRODUCT": "#cce5ff",
                    "EVENT":   "#ffeeba",
                    "NORP":    "#f5c6cb",
                }
                ENTITY_ICONS = {
                    "PERSON":  "👤",
                    "ORG":     "🏢",
                    "GPE":     "🌍",
                    "LOC":     "📍",
                    "MONEY":   "💰",
                    "PERCENT": "📊",
                    "DATE":    "📅",
                    "PRODUCT": "📦",
                    "EVENT":   "🎯",
                    "NORP":    "🏛️",
                }
                for label, items in entities.items():
                    icon  = ENTITY_ICONS.get(label, "🔹")
                    color = ENTITY_COLORS.get(label, "#f0f0f0")
                    tags  = "".join(
                        f'<span style="background:{color};border-radius:20px;'
                        f'padding:3px 10px;margin:3px;display:inline-block;font-size:0.85rem;">'
                        f'{item}</span>'
                        for item in items
                    )
                    st.markdown(f"**{icon} {label}**", unsafe_allow_html=True)
                    st.markdown(tags, unsafe_allow_html=True)
            else:
                st.info("No entities found.")

        with col5:
            st.markdown('<div class="section-header">🔑 Top Keywords</div>', unsafe_allow_html=True)
            if keywords:
                kw_fig = go.Figure(go.Bar(
                    x=[k["score"] for k in keywords],
                    y=[k["keyword"] for k in keywords],
                    orientation="h",
                    marker_color="#17becf"
                ))
                kw_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(kw_fig, use_container_width=True)
            else:
                st.info("No keywords found.")

        # ── Row 3: Sentiment Breakdown ─────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🎭 Sentiment Breakdown</div>', unsafe_allow_html=True)
        s_scores = sentiment["scores"]
        sent_fig = go.Figure(go.Bar(
            x=list(s_scores.keys()),
            y=list(s_scores.values()),
            marker_color=["#28a745", "#dc3545", "#6c757d"]
        ))
        sent_fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            yaxis=dict(range=[0, 100]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(sent_fig, use_container_width=True)