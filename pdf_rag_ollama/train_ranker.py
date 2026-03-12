import json
import os
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

def train_ranker(data_path: str, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", output_path: str = "fine-tuned-ranker"):
    # Load training data
    with open(data_path, "r") as f:
        data = json.load(f)

    train_examples = []
    for item in data:
        query = item["query"]
        for text, label in zip(item["texts"], item["labels"]):
            train_examples.append(InputExample(texts=[query, text], label=float(label)))

    # Define model
    model = CrossEncoder(model_name, num_labels=1)

    # Prepare data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Train (briefly for demonstration)
    print(f"Starting training on {len(train_examples)} examples...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        loss_fct=torch.nn.BCEWithLogitsLoss(),
        optimizer_params={'lr': 2e-5}
    )

    # Save model
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_ranker("training_data.json")
