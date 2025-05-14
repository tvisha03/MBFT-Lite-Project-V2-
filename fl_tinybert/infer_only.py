import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============ CONFIG ============
CLIENT_ID = 4
BASE_PATH = Path("./saved_models").resolve()
MODEL_PATH = BASE_PATH / f"client_{CLIENT_ID}"
DATA_PATH = Path(f"../data/processed/text/client_{CLIENT_ID}_text.csv").resolve()
SAVE_PATH = Path(f"./fl_tinybert/results/client_{CLIENT_ID}/predictions.csv").resolve()
# ================================

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.to(device)
model.eval()

# Prepare Dataset
class TinyBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load data and split
df = pd.read_csv(DATA_PATH)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, stratify=df["label"], random_state=42
)

dataset = TinyBERTDataset(test_texts, test_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=32)

# Inference loop
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)

# Save results
os.makedirs(SAVE_PATH.parent, exist_ok=True)
df_out = pd.DataFrame({
    "Actual": all_labels,
    "Predicted": all_preds,
    "Probability": all_probs
})
df_out.to_csv(SAVE_PATH, index=False)
print(f"✅ Saved TinyBERT predictions with probabilities for Client {CLIENT_ID} → {SAVE_PATH}")
