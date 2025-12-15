from huggingface_hub import login

@st.cache_resource
def load_model():
    hf_token = os.environ.get("HF_TOKEN")  # populated from Streamlit secrets
    if not hf_token:
        raise ValueError("HF_TOKEN not set in environment / secrets")

    login(token=hf_token)

    model_id = "korgtmf/FOMO"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=hf_token)
    return tokenizer, model
