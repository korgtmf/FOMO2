from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from evaluate import load
import numpy as np
import torch

# 1. Load sentiment model & tokenizer (FinBERT tone, 3 classes)
base_model_id = "yiyanghkust/finbert-tone"  # positive / negative / neutral
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# (Optional) If you really want 3 classes, keep num_labels=3
config = AutoConfig.from_pretrained(base_model_id)
config.num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id,
    config=config,
    ignore_mismatched_sizes=True,
)

# 2. Load Yelp dataset and map 5 stars -> 3 sentiment classes
raw = load_dataset("yelp_review_full")  # splits: train, test

def map_to_3classes(example):
    star = example["label"]  # 0..4 (1..5 stars)
    if star <= 1:
        return {"label": 0}  # negative
    elif star == 2:
        return {"label": 1}  # neutral
    else:
        return {"label": 2}  # positive

raw = raw.map(map_to_3classes)

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_datasets = raw.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# keep "text" if you want, but Trainer only needs tensors
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])

tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 4. Metric
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. Training arguments (sentiment + epochs)
training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=3,
    eval_strategy="epoch",   # if your version supports eval_strategy
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
