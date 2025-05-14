import os
import torch
import flwr as fl
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Constants
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
NUM_LABELS = 2
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3

client_id = int(os.getenv("CLIENT_ID", "1"))
csv_path = f"../data/processed/text/client_{client_id}_text.csv"


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

# Load and tokenize
def load_data():
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"label": "labels"})  # Ensure compatibility
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "__index_level_0__"]) if "__index_level_0__" in tokenized_dataset.column_names else tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    return split["train"], split["test"]

train_dataset, eval_dataset = load_data()

# Metric function
def compute_metrics(pred):
    preds = torch.argmax(torch.tensor(pred.predictions), dim=1)
    labels = torch.tensor(pred.label_ids)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Load model
def get_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Federated client class
class TinyBertClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        training_args = TrainingArguments(
            output_dir=f"./client_{client_id}_tinybert",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            evaluation_strategy="no",
            logging_strategy="no",
            save_strategy="no",
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        # SAVE MODEL AFTER TRAINING
        save_dir = f"./saved_models/client_{client_id}"
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[Client {client_id}] ✅ Model saved to {save_dir}")

        return self.get_parameters(), len(train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=f"./client_{client_id}_eval",
                per_device_eval_batch_size=BATCH_SIZE,
                disable_tqdm=True
            ),
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        eval_result = trainer.evaluate()
        # SAVE EVALUATION RESULTS AND PREDICTIONS
        preds = trainer.predict(eval_dataset)
        y_pred = np.argmax(preds.predictions, axis=1)
        y_true = preds.label_ids

        results_dir = f"./results/client_{client_id}"
        os.makedirs(results_dir, exist_ok=True)

        # Save predictions
        pd.DataFrame({
        "true": y_true,
        "pred": y_pred
        }).to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

        # Save metrics
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(eval_result, f, indent=4)

        print(f"[Client {client_id}] 📊 Predictions and metrics saved to {results_dir}")

        return float(eval_result["eval_loss"]), len(eval_dataset), eval_result

# Launch
if __name__ == "__main__":
    print(f"🚀 Starting TinyBERT Client {client_id}")
    fl.client.start_numpy_client(server_address="localhost:8080", client=TinyBertClient())
