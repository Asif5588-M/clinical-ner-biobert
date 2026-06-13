import json
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import classification_report, f1_score
import torch

PROCESSED   = Path("data/processed")
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.2"
OUTPUT_DIR  = str(MODELS_DIR / "clinical-ner-biobert")

print("=" * 60)
print("Clinical NER — BioBERT Fine-tuning")
print("=" * 60)

# ── Load Data ──────────────────────────────────────────────
with open(PROCESSED / "label_map.json") as f:
    label_map = json.load(f)

label2id = label_map["label2id"]
id2label = {int(k): v for k, v in label_map["id2label"].items()}
NUM_LABELS = len(label2id)

print(f"\nLabels ({NUM_LABELS}): {list(label2id.keys())}")

def load_split(name):
    with open(PROCESSED / f"{name}.json") as f:
        data = json.load(f)
    return {"tokens": [d["tokens"] for d in data],
            "labels": [d["labels"] for d in data]}

train_data = load_split("train")
val_data   = load_split("val")
test_data  = load_split("test")

print(f"Train: {len(train_data['tokens'])}")
print(f"Val  : {len(val_data['tokens'])}")
print(f"Test : {len(test_data['tokens'])}")

dataset = DatasetDict({
    "train"     : Dataset.from_dict(train_data),
    "validation": Dataset.from_dict(val_data),
    "test"      : Dataset.from_dict(test_data),
})

# ── Tokenizer ──────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation     = True,
        max_length     = 128,
        is_split_into_words = True,
        padding        = "max_length",
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids     = tokenized.word_ids(batch_index=i)
        label_ids    = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[labels[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_and_align, batched=True)
print("Tokenization done ✓")

# ── Model ──────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels = NUM_LABELS,
    id2label   = id2label,
    label2id   = label2id,
    ignore_mismatched_sizes = True,
)
print("Model loaded ✓")

# ── Metrics ────────────────────────────────────────────────
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [id2label[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "f1"       : f1_score(true_labels, true_preds),
        "precision": report.get("weighted avg", {}).get("precision", 0),
        "recall"   : report.get("weighted avg", {}).get("recall", 0),
    }

# ── Training Args ──────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = 5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    warmup_steps            = 100,
    weight_decay            = 0.01,
    learning_rate           = 2e-5,
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1",
    logging_steps           = 50,
    fp16                    = torch.cuda.is_available(),
    report_to               = "none",
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_dataset["train"],
    eval_dataset    = tokenized_dataset["validation"],
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

# ── Train ──────────────────────────────────────────────────
print("\nStarting training...")
print("=" * 60)
trainer.train()

# ── Evaluate ───────────────────────────────────────────────
print("\nEvaluating on test set...")
results = trainer.evaluate(tokenized_dataset["test"])
print(f"\nTest Results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ── Save ───────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved: {OUTPUT_DIR}")
print("\nNext → Upload to HuggingFace Hub")