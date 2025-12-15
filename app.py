# app.py - Rule-based Sentiment + Topic Matching Streamlit App
import streamlit as st
import re

st.set_page_config(page_title="Sentiment & Topic Analyzer", page_icon="😊", layout="wide")

st.title("😊 Sentiment & Topic Analyzer")
st.markdown("**Zero dependencies – sentiment + topic matching.**")

# ---------- Sentiment Analysis (rule-based) ----------
def analyze_sentiment(text: str):
    text = text.lower()

    positive_words = {
        'great': 1.2, 'excellent': 1.5, 'amazing': 1.4, 'love': 1.3, 'perfect': 1.5,
        'good': 1.0, 'fantastic': 1.3, 'wonderful': 1.2, 'awesome': 1.3, 'best': 1.4,
        'happy': 1.1, 'recommend': 1.2, 'success': 1.3, 'profit': 1.2, 'growth': 1.1
    }
    negative_words = {
        'bad': -1.0, 'terrible': -1.5, 'awful': -1.4, 'hate': -1.3, 'worst': -1.5,
        'poor': -1.1, 'horrible': -1.4, 'disappointed': -1.2, 'fail': -1.3, 'loss': -1.2,
        'slow': -0.8, 'expensive': -1.0, 'broken': -1.2, 'scam': -1.5
    }

    score = 0
    hits_pos, hits_neg = [], []

    for w, wt in positive_words.items():
        if w in text:
            score += wt
            hits_pos.append(w)

    for w, wt in negative_words.items():
        if w in text:
            score += wt
            hits_neg.append(w)

    total_hits = max(1, len(hits_pos) + len(hits_neg))
    avg_score = score / total_hits

    if avg_score > 0.5:
        label = "✅ Positive"
    elif avg_score < -0.3:
        label = "❌ Negative"
    else:
        label = "😐 Neutral"

    confidence = max(40, min(95, 60 + avg_score * 20))
    return label, round(confidence), hits_pos, hits_neg

# ---------- Topic Matching (rule-based) ----------
TOPIC_KEYWORDS = {
    "Stock Market": [
        "stock", "share", "equity", "ipo", "dividend", "index", "nasdaq", "dow", "s&p"
    ],
    "Cryptocurrency": [
        "bitcoin", "ethereum", "crypto", "token", "blockchain", "defi", "nft"
    ],
    "Banking": [
        "bank", "loan", "interest rate", "mortgage", "deposit", "credit card", "lending"
    ],
    "Investment": [
        "portfolio", "invest", "fund", "etf", "asset allocation", "hedge fund", "long-term"
    ],
    "Economy": [
        "inflation", "gdp", "unemployment", "recession", "macro", "economic", "central bank"
    ],
    "Corporate Finance": [
        "earnings", "revenue", "profit", "loss", "merger", "acquisition", "buyback", "guidance"
    ],
    "Personal Finance": [
        "saving", "budget", "retirement", "pension", "salary", "credit score", "debt"
    ],
    "Customer Service": [
        "service", "support", "staff", "waiter", "response time", "customer", "complaint"
    ]
}

def match_topics(text: str):
    text_low = text.lower()
    topic_scores = {}
    topic_hits = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        hits = []
        for kw in keywords:
            if kw in text_low:
                score += 1
                hits.append(kw)
        if score > 0:
            topic_scores[topic] = score
            topic_hits[topic] = hits

    if not topic_scores:
        return "General", {}, {}

    max_score = max(topic_scores.values())
    best_topics = [t for t, s in topic_scores.items() if s == max_score]

    best_topic = best_topics[0]
    # Normalize relevance 0–100
    relevance = {t: int(100 * s / max_score) for t, s in topic_scores.items()}

    return best_topic, relevance, topic_hits

# ---------- UI: Samples ----------
col1, col2, col3 = st.columns(3)
samples = [
    "The company reported excellent earnings and strong profit growth this quarter.",
    "Bitcoin price fell sharply as regulators announced new rules.",
    "Customer service was terrible, I am very disappointed with this bank."
]

with col1:
    if st.button("📈 Earnings sample", use_container_width=True):
        st.session_state.text = samples[0]
with col2:
    if st.button("🪙 Crypto sample", use_container_width=True):
        st.session_state.text = samples[1]
with col3:
    if st.button("🏦 Banking CX sample", use_container_width=True):
        st.session_state.text = samples[2]

# ---------- Main input ----------
text = st.text_area(
    "📝 Enter text:",
    height=150,
    placeholder="Any review, headline or financial text...",
    key="text"
)

if st.button("🔍 Analyze Sentiment & Topic", type="primary", use_container_width=True) and text.strip():
    sent_label, sent_conf, hits_pos, hits_neg = analyze_sentiment(text)
    topic_label, topic_relevance, topic_hits = match_topics(text)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"### {sent_label}")
        st.metric("Sentiment confidence", f"{sent_conf}%")
        st.caption(f"*\"{text[:140]}{'...' if len(text) > 140 else ''}\"*")

    with c2:
        st.markdown("**Main topic**")
        st.metric("Topic", topic_label)

    with c3:
        top_rel = topic_relevance.get(topic_label, 100)
        st.metric("Topic relevance", f"{top_rel}%")

    st.markdown("### 📊 Topic breakdown")
    if topic_relevance:
        topic_rows = [{"Topic": t, "Relevance %": r} for t, r in sorted(topic_relevance.items(), key=lambda x: -x[1])]
        st.dataframe(topic_rows, use_container_width=True)
    else:
        st.info("No specific topic detected; classified as **General**.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Positive words")
        if hits_pos:
            for w in sorted(set(hits_pos)):
                st.caption(f"• {w}")
        else:
            st.caption("• None detected")

    with col_b:
        st.subheader("❌ Negative words")
        if hits_neg:
            for w in sorted(set(hits_neg)):
                st.caption(f"• {w}")
        else:
            st.caption("• None detected")

    st.subheader("🏷 Topic keywords hit")
    if topic_hits:
        for t, kws in topic_hits.items():
            st.markdown(f"**{t}**: " + ", ".join(sorted(set(kws))))
    else:
        st.caption("No topic-specific keywords triggered.")

# ---------- Batch mode ----------
st.markdown("---")
st.subheader("📦 Batch sentiment + topic")

batch_text = st.text_area("One text per line:", height=150, key="batch")
max_rows = st.slider("Max rows to display", 1, 50, 10, key="max_rows")

if st.button("🚀 Analyze batch", use_container_width=True) and batch_text.strip():
    rows = []
    for line in [l.strip() for l in batch_text.splitlines() if l.strip()][:max_rows]:
        s_label, s_conf, _, _ = analyze_sentiment(line)
        t_label, _, _ = match_topics(line)
        rows.append({
            "Text preview": line[:40] + ("..." if len(line) > 40 else ""),
            "Sentiment": s_label.split()[1],
            "Conf %": s_conf,
            "Topic": t_label
        })
    st.dataframe(rows, use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 What it does")
    st.markdown("- Rule-based sentiment (pos/neg/neutral).[web:1][web:3]")
    st.markdown("- Rule-based topic matching for finance & CX.[web:5]")
    st.markdown("- No model downloads or external calls.")
    st.markdown("---")
    st.caption("Prototype: adjust keywords per domain for better recall.")
