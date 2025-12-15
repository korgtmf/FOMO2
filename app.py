import streamlit as st
import os
from huggingface_hub import login
from transformers import pipeline
import torch

@st.cache_resource
def load_model():
    os.environ["HF_TOKEN"] = st.secrets.get("HF_TOKEN", "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")
    login(token=os.environ["HF_TOKEN"])
    
    pipe = pipeline(
        "text-classification",
        model="korgtmf/FOMO",
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe

pipe = load_model()
st.write("Model loaded successfully")
