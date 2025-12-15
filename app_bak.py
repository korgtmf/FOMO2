# app.py - FinBERT Sentiment Analysis Web App
from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import re
from collections import Counter
import os

app = Flask(__name__)

# Load FinBERT model (3-class: negative/neutral/positive)
model_id = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Star-to-sentiment mapping (for Yelp-style input)
STAR_MAPPING = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2}  # neg/neut/pos

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Please enter text'}), 400
    
    # Get predictions
    predictions = classifier(text)
    
    # Format results
    scores = {pred['label']: round(pred['score'] * 100, 1) for pred in predictions[0]}
    top_pred = max(scores, key=scores.get)
    
    # Sentiment labels
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    return jsonify({
        'prediction': labels[int(top_pred.split('_')[-1])],
        'confidence': scores[top_pred],
        'scores': {
            'Negative': scores.get('LABEL_0', 0),
            'Neutral': scores.get('LABEL_1', 0), 
            'Positive': scores.get('LABEL_2', 0)
        },
        'text': text[:100] + '...' if len(text) > 100 else text
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    results = []
    for text in texts[:10]:  # Limit to 10 for demo
        if text.strip():
            preds = classifier(text.strip())
            top_pred = max(preds[0], key=lambda x: x['score'])
            results.append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'sentiment': top_pred['label'].split('_')[-1],
                'confidence': round(top_pred['score'] * 100, 1)
            })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
