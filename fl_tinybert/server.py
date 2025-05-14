# fl_tinybert/server.py

import flwr as fl
from transformers import AutoModelForSequenceClassification

# Load TinyBERT model architecture
def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D",
        num_labels=2,
    )
    return model

# Define the FL strategy (FedAvg for now)
strategy = fl.server.strategy.FedAvg()

if __name__ == "__main__":
    print("🚀 Federated TinyBERT Server starting...")
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
