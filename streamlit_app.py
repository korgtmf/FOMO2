import streamlit as st
import os
from huggingface_hub import login
from transformers import pipeline
import torch

# Page config
st.set_page_config(page_title="Financial News Topic Classifier", layout="wide")

# HF Token (use Streamlit secrets for production)
@st.cache_resource
def load_model():
    os.environ["HF_TOKEN"] = st.secrets.get("HF_TOKEN", "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")
    login(token=os.environ["HF_TOKEN"])
    
    pipe = pipeline(
        "text-classification",
        model="korgtmf/FOMO",
        truncation=True,
        device=0 if torch.cuda.is_available() else -1  # GPU if available
    )
    return pipe

# Load model once
pipe = load_model()

# Topic mapping (for reference)
topics = {
    0: "Analyst Update", 1: "Fed | Central Banks", 2: "Company | Product News",
    3: "Treasuries | Corporate Debt", 4: "Dividend", 5: "Earnings", 6: "Energy | Oil",
    7: "Financials", 8: "Market Commentary", 9: "Macro", 10: "Mergers & Acquisitions",
    11: "Metals | Materials", 12: "Price Target", 13: "Public Offerings",
    14: "Stock Commentary", 15: "Stock Movement", 16: "Tech", 17: "Top News",
    18: "Wall Street | Insider Trading", 19: "Other"
}

def get_topic_name(label_str: str) -> str:
    if label_str.startswith("LABEL_"):
        idx = int(label_str.replace("LABEL_", ""))
        return topics.get(idx, f"Unknown topic (index {idx})")
    return label_str

# Header
st.title("📰 Financial News Topic Classifier")
st.markdown("Classify financial tweets/news using FOMO model (20 topics)")

# Sidebar examples
st.sidebar.header("Example Inputs")
examples = [
    "AAPL beats Q4 earnings by 10%",
    "Fed signals 50bps rate cut", 
    "MSFT acquires GitHub for $7.5B",
    "Oil surges to $90/barrel",
    "TSLA price target raised to $350"
]
selected_example = st.sidebar.selectbox("Quick test:", examples)
st.sidebar.info(f"**Model**: korgtmf/FOMO\n**Topics**: 20 financial categories")

# Main input
col1, col2 = st.columns([3, 1])
with col1:
    user_text = st.text_area(
        "Enter financial news/tweet:",
        value=selected_example or "",
        height=100,
        placeholder="e.g. 'NVDA announces new AI chip...'",
        help="20-140 chars work best"
    )
with col2:
    st.markdown("### Confidence")
    confidence_placeholder = st.empty()

# Classify button
if st.button("🔍 Classify Topic", type="primary"):
    if user_text.strip():
        with st.spinner("Classifying..."):
            result = pipe(user_text)[0]
            topic_name = get_topic_name(result["label"])
            score = result["score"]
            
            # Results
            st.success(f"**Predicted: {topic_name}**")
            st.metric("Confidence", f"{score:.1%}", delta=None)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.json({"label": result["label"], "score": score})
            with col_b:
                st.caption(f"**Raw score**: {score:.4f}")
                
        confidence_placeholder.metric("Confidence", f"{score:.1%}")
    else:
        st.warning("Enter some financial news text!")

# Batch mode
st.header("📊 Batch Classification")
batch_text = st.text_area("Multiple texts (one per line):", height=150)
if st.button("Classify Batch") and batch_text.strip():
    texts = [t.strip() for t in batch_text.split("\n") if t.strip()]
    if texts:
        results_df = []
        for text in texts:
            result = pipe(text)[0]
            results_df.append({
                "Text": text[:60] + "..." if len(text) > 60 else text,
                "Topic": get_topic_name(result["label"]),
                "Confidence": f"{result['score']:.1%}"
            })
        st.dataframe(results_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Powered by [korgtmf/FOMO](https://huggingface.co/korgtmf/FOMO) • Optimized for fintech news")
