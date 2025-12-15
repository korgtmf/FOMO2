import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


@st.cache_resource
def load_model():
    model_id = "korgtmf/FOMO"  # or your exact HF repo id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model
