import json
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from VBERTClassifier import VBERTClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from MMHS150K import MMHS150KDataset
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VBERTClassifier().to(device)

# Weights to counter data imbalance and avoid overfitting to one class
weights = torch.tensor([0.204, 1.929, 6.586, 5.949, 141.22, 3.963], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.AdamW(model.parameters(), lr=2e-5)


def collate_fn(batch):
    """
    Custom collate function to combine individual samples into batched tensors.
    It handles stacking of text tokens, labels, and concatenating the visual features
    which already include a batch dimension.
    """
    # Stack text tokens and labels
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    # Concatenate visual embeddings and attention masks (they are of shape [1, 36, D] and [1, 36])
    visual_embeds = torch.cat([item["visual_embeds"] for item in batch], dim=0)
    visual_attention_mask = torch.cat(
        [item["visual_attention_mask"] for item in batch], dim=0
    )

    # Concatenate normalized boxes (they are of shape [1, 36, 4])
    norm_boxes = torch.cat([item["norm_boxes"] for item in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "visual_embeds": visual_embeds,
        "visual_attention_mask": visual_attention_mask,
        "norm_boxes": norm_boxes,
        "labels": labels,
    }


def train(model, epochs=5, version_num=1):
    version = f"v{version_num}"
    save_dir = f"./models/{version}/"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            start_time = time.time()

            # Data loading and transferring to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            visual_embeds = batch["visual_embeds"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            forward_start = time.time()
            logits = model(input_ids, attention_mask, visual_embeds)
            forward_time = time.time() - forward_start

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start

            batch_time = time.time() - start_time

            # Update progress bar
            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))
            if (batch_idx + 1) % 100 == 0:
                tqdm.write(
                    f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | "
                    f"Batch Time: {batch_time:.3f}s | Forward: {forward_time:.3f}s | Backward: {backward_time:.3f}s"
                )

            # Save the model every 1000 iterations
            if (batch_idx + 1) % 1000 == 0:
                checkpoint_path = os.path.join(
                    save_dir, f"vbert_{version}_epoch{epoch+1}_iter{batch_idx+1}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                tqdm.write(
                    f"Saved model checkpoint at iteration {batch_idx+1} to {checkpoint_path}"
                )

        # Save model after every epoch
        epoch_checkpoint_path = os.path.join(
            save_dir, f"vbert_{version}_epoch{epoch+1}.pt"
        )
        torch.save(model.state_dict(), epoch_checkpoint_path)
        tqdm.write(
            f"Saved model checkpoint for epoch {epoch+1} to {epoch_checkpoint_path}"
        )


def evaluate_model(model, data_loader, num_classes=6, plot_confusion_matrix=True):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_true = []
    all_preds = []

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for batch in progress_bar:
            # Transfer batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            visual_embeds = batch["visual_embeds"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass through the model
            logits = model(input_ids, attention_mask, visual_embeds)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_true.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            progress_bar.set_postfix({"Acc": total_correct / total_samples})

    overall_accuracy = total_correct / total_samples
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    # Compute detailed classification metrics
    report = classification_report(
        all_true, all_preds, target_names=[f"Class {i}" for i in range(num_classes)]
    )
    print("\nClassification Report:\n", report)

    # Compute and plot confusion matrix.
    cm = confusion_matrix(all_true, all_preds)
    if plot_confusion_matrix:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Pred {i}" for i in range(num_classes)],
            yticklabels=[f"True {i}" for i in range(num_classes)],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    faster_rcnn = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    ).to(device)
    faster_rcnn.eval()

    # File paths
    MMHS150K_PATH = "../data/MMHS150K/"
    DATASET_PATH = f"{MMHS150K_PATH}/MMHS150K_GT.json"
    IMG_DIR = f"{MMHS150K_PATH}/img_resized"
    OCR_DIR = f"{MMHS150K_PATH}/img_txt/"
    TRAIN_SPLIT = f"{MMHS150K_PATH}/splits/train_ids.txt"
    VAL_SPLIT = f"{MMHS150K_PATH}/splits/val_ids.txt"
    TEST_SPLIT = f"{MMHS150K_PATH}/splits/test_ids.txt"

    # Load dataset
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    # Load train/val splits
    with open(TRAIN_SPLIT, "r") as f:
        train_ids = set(f.read().splitlines())
    with open(VAL_SPLIT, "r") as f:
        val_ids = set(f.read().splitlines())
    with open(TEST_SPLIT, "r") as f:
        test_ids = set(f.read().splitlines())

    train_dataset = MMHS150KDataset(
        train_ids, data, tokenizer, IMG_DIR, OCR_DIR, faster_rcnn, device
    )
    val_dataset = MMHS150KDataset(
        val_ids, data, tokenizer, IMG_DIR, OCR_DIR, faster_rcnn, device
    )
    test_dataset = MMHS150KDataset(
        test_ids, data, tokenizer, IMG_DIR, OCR_DIR, faster_rcnn, device
    )

    num_workers = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # pin_memory=True,
    )

    # Load the checkpoint
    checkpoint_path = "./models/v1/vbert_v1_epoch1.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Run validation
    model.eval()
    evaluate_model(model, val_loader)

    # model.train()
    # train(model, epochs=1)
