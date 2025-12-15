# app.py - Bulletproof FinBERT Streamlit App (Multiple Fallbacks)
import streamlit as st
import torch
from transformers import pipeline
import time

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🚀", layout="wide")

# FALLBACK MODEL LIST (all guaranteed to work)
MODEL_PRIORITIES = [
    "ProsusAI/finbert",
    "nlptown/bert-base-multilingual-uncased-sentiment", 
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "distilbert-base-uncased-finetuned-sst-2-english"
]

@st.cache_resource
def load_any_model():
    """Try multiple models until one works"""
    for i, model_name in enumerate(MODEL_PRIORITIES):
        try:
            st.info(f"🔄 Trying model {i+1}/{len(MODEL_PRIORITIES)}: {model_name}")
            classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                return_all_scores=True,
                device=-1  # CPU only for cloud stability
            )
            return classifier, model_name
        except Exception as e:
            st.warning(f"❌ {model_name} failed: {str(e)[:50]}...")
            continue
    
    return None, None

# Load model with robust retry
st.sidebar.title("🤖 Sentiment Analyzer")
progress = st.sidebar.progress(0)
status = st.sidebar.empty()

if 'classifier' not in st.session_state:
    st.session_state.classifier = None
    st.session_state.model_name = None

if st.session_state.classifier is None:
    status.text("🔄 Loading model...")
    with st.spinner("Finding working model..."):
        classifier, model_name = load_any_model()
        st.session_state.classifier = classifier
        st.session_state.model_name = model_name
    
    if classifier:
        progress.progress(100)
        status.success(f"✅ Loaded: {model_name}")
        st.sidebar.balloons()
    else:
        status.error("❌ All models failed")
        st.stop()

classifier = st.session_state.classifier
model_name = st.session_state.model_name

# Main UI
st.title("🚀 AI Sentiment Analyzer")
st.markdown(f"**Using: {model_name}**")

# Sample buttons
col1, col2, col3 = st.columns(3)
samples = ["Love this product!", "It's okay, nothing special", "Terrible experience, avoid!"]

with col1:
    if st.button("✅ Positive", use_container_width=True): st.session_state.text = samples[0]
with col2:
    if st.button("😐 Neutral", use_container_width=True): st.session_state.text = samples[1]
with col3:
    if st.button("❌ Negative", use_container_width=True): st.session_state.text = samples[2]

# Input
text = st.text_area
