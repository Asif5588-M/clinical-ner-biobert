import json
import pandas as pd
import re
from pathlib import Path
from collections import Counter
import random

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Clinical NER — Combined Data Preprocessing")
print("=" * 60)

# ── Load Corona2.json ──────────────────────────────────────
with open(RAW / "Corona2.json", "r") as f:
    corona_data = json.load(f)

def convert_to_bio(text, annotations):
    tokens = text.split()
    labels = ["O"] * len(tokens)
    char_to_token = {}
    char_pos = 0
    for i, token in enumerate(tokens):
        for j in range(len(token)):
            char_to_token[char_pos + j] = i
        char_pos += len(token) + 1
    for ann in annotations:
        start = ann.get("start", 0)
        end   = ann.get("end", 0)
        tag   = ann.get("tag_name", "O")
        if not tag or tag == "O":
            continue
        token_start = char_to_token.get(start)
        token_end   = char_to_token.get(end - 1)
        if token_start is None or token_end is None:
            continue
        labels[token_start] = f"B-{tag}"
        for k in range(token_start + 1, token_end + 1):
            if k < len(labels):
                labels[k] = f"I-{tag}"
    return tokens, labels

corona_processed = []
for ex in corona_data["examples"]:
    text = ex["content"]
    anns = ex["annotations"]
    if not text.strip():
        continue
    tokens, labels = convert_to_bio(text, anns)
    if tokens:
        corona_processed.append({"tokens": tokens, "labels": labels})

print(f"\nCorona2 examples : {len(corona_processed)}")

# ── Load MTSamples ─────────────────────────────────────────
df = pd.read_csv(RAW / "mtsamples.csv")
print(f"MTSamples rows   : {len(df)}")
print(f"Columns          : {df.columns.tolist()}")

# Use transcription column
df = df.dropna(subset=["transcription"])
df = df[df["transcription"].str.len() > 100]
print(f"After cleaning   : {len(df)} rows")

# ── Medical terms for rule-based tagging ──────────────────
MEDICINES = {
    "aspirin","ibuprofen","metformin","insulin","amoxicillin",
    "lisinopril","atorvastatin","metoprolol","omeprazole","amlodipine",
    "warfarin","acetaminophen","prednisone","levothyroxine","albuterol",
    "hydrocodone","gabapentin","sertraline","furosemide","pantoprazole",
    "clopidogrel","zolpidem","oxycodone","tramadol","ciprofloxacin",
    "doxycycline","azithromycin","clindamycin","vancomycin","heparin",
    "morphine","lorazepam","diazepam","metronidazole","fluoxetine",
    "simvastatin","ramipril","carvedilol","digoxin","tamsulosin",
}

CONDITIONS = {
    "diabetes","hypertension","pneumonia","cancer","tumor","infection",
    "fever","pain","nausea","vomiting","diarrhea","cough","dyspnea",
    "fatigue","headache","depression","anxiety","obesity","asthma",
    "stroke","myocardial","infarction","fracture","sepsis","anemia",
    "hypertensive","diabetic","chronic","acute","bilateral","carcinoma",
    "edema","hemorrhage","thrombosis","stenosis","arrhythmia","seizure",
    "dementia","alzheimer","parkinson","arthritis","osteoporosis","fibrosis",
}

PATHOGENS = {
    "covid","sars","influenza","hiv","hepatitis","tuberculosis",
    "salmonella","staphylococcus","streptococcus","pneumococcal",
    "mrsa","ecoli","candida","aspergillus","pseudomonas","clostridium",
}

def tag_sentence(tokens):
    labels = []
    prev_label = "O"
    for token in tokens:
        t = token.lower().strip(".,;:!?()")
        if t in MEDICINES:
            label = "B-Medicine" if prev_label != "I-Medicine" else "I-Medicine"
        elif t in CONDITIONS:
            label = "B-MedicalCondition" if prev_label != "I-MedicalCondition" else "I-MedicalCondition"
        elif t in PATHOGENS:
            label = "B-Pathogen" if prev_label != "I-Pathogen" else "I-Pathogen"
        else:
            label = "O"
        labels.append(label)
        prev_label = label
    return labels

# ── Process MTSamples sentences ────────────────────────────
mt_processed = []
for _, row in df.iterrows():
    text = str(row["transcription"])
    sentences = re.split(r'[.!?]\s+', text)
    for sent in sentences:
        sent = sent.strip()
        words = sent.split()
        if len(words) < 5 or len(words) > 80:
            continue
        labels = tag_sentence(words)
        # Only keep sentences with at least one entity
        if any(l != "O" for l in labels):
            mt_processed.append({"tokens": words, "labels": labels})
        if len(mt_processed) >= 2000:
            break
    if len(mt_processed) >= 2000:
        break

print(f"MTSamples NER    : {len(mt_processed)} sentences")

# ── Combine ────────────────────────────────────────────────
all_data = corona_processed + mt_processed
random.seed(42)
random.shuffle(all_data)
print(f"\nTotal combined   : {len(all_data)} examples")

# ── Label distribution ─────────────────────────────────────
all_labels = []
for p in all_data:
    all_labels.extend(p["labels"])
label_counts = Counter(all_labels)
print(f"\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label:<30} {count:>6}")

# ── Train/Val/Test split ───────────────────────────────────
n         = len(all_data)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
train     = all_data[:train_end]
val       = all_data[train_end:val_end]
test      = all_data[val_end:]

print(f"\nSplit:")
print(f"  Train : {len(train)}")
print(f"  Val   : {len(val)}")
print(f"  Test  : {len(test)}")

# ── Save ───────────────────────────────────────────────────
for name, data in [("train",train),("val",val),("test",test)]:
    with open(PROCESSED / f"{name}.json", "w") as f:
        json.dump(data, f)

unique_labels = sorted(set(all_labels))
label2id = {l: i for i, l in enumerate(unique_labels)}
id2label  = {i: l for l, i in label2id.items()}
with open(PROCESSED / "label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

print(f"\n{'='*60}")
print("Saved to data/processed/")
print(f"Total labels     : {len(unique_labels)}")
print(f"Labels           : {unique_labels}")
print(f"{'='*60}")
print("\nNext → src/train_ner.py")