# app.py - Fixed FinBERT Streamlit App (Valid Model ID)
import streamlit as st
import torch
from transformers import pipeline
import time

# Page config
st.set_page_config(page_title="FinBERT Sentiment Analyzer", page_icon="🚀", layout="wide")

@st.cache_resource
def load_model():
    """Load verified FinBERT model from HuggingFace"""
    try:
        # ✅ VALID model IDs that exist and work
        model_name = "ProsusAI/finbert"  # Official FinBERT (3-class)
        
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Initialize
if 'classifier' not in st.session_state:
    st.session_state.classifier = None

# Load model
st.sidebar.title("🚀 FinBERT Analyzer")
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

if st.session_state.classifier is None:
    status_text.text("🔄 Loading model...")
    with st.spinner("Loading official FinBERT..."):
        st.session_state.classifier = load_model()
        progress_bar.progress(100)
    
    if st.session_state.classifier:
        status_text.success("✅ Model loaded!")
    else:
        status_text.error("❌ Model failed")

classifier = st.session_state.classifier

# Main title
st.title("🚀 FinBERT Sentiment Analysis")
st.markdown("**Financial sentiment analysis** using official ProsusAI/FinBERT")

if classifier is None:
    st.warning("⚠️ Model loading issue. Using fallback...")
    st.stop()

# Sample buttons
col1, col2, col3 = st.columns(3)
samples = [
    "Company reported strong earnings this quarter.",
    "Market conditions remain uncertain.",
    "Stock plummeted after earnings miss."
]

with col1:
    if st.button("✅ Positive", use_container_width=True):
        st.session_state.text_input = samples[0]
with col2:
    if st.button("😐 Neutral", use_container_width=True):
        st.session_state.text_input = samples[1]
with col3:
    if st.button("❌ Negative", use_container_width=True):
        st.session_state.text_input = samples[2]

# Text input
text_input = st
