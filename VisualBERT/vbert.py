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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VBERTClassifier().to(device)

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


def train(model, epochs=5):
    # Ensure the save directory exists.
    version = "v3"
    save_dir = f"./{version}/"
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
            visual_attention_mask = batch["visual_attention_mask"].to(device)
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

            # Update progress bar (this will update on a single line)
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


# Validation step
def validate(model, val_loader):
    # Initialize counters for overall and per-class accuracy.
    num_classes = 6
    total_samples = 0
    total_correct = 0
    total_per_class = [0] * num_classes
    correct_per_class = [0] * num_classes

    # Create a tqdm progress bar over the validation DataLoader.
    progress_bar = tqdm(val_loader, desc="Validating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            # Transfer batch data to GPU.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            visual_embeds = batch["visual_embeds"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass.
            logits = model(input_ids, attention_mask, visual_embeds)
            predictions = torch.argmax(logits, dim=1)

            # Update overall accuracy counters.
            total_samples += labels.size(0)
            total_correct += (predictions == labels).sum().item()

            # Update per-class counters.
            for i in range(num_classes):
                mask = labels == i
                total_per_class[i] += mask.sum().item()
                if mask.sum().item() > 0:
                    correct_per_class[i] += (
                        (predictions[mask] == labels[mask]).sum().item()
                    )

            # Update progress bar with current overall accuracy.
            progress_bar.set_postfix({"Overall Acc": total_correct / total_samples})

    overall_accuracy = total_correct / total_samples
    print(f"\nOverall Validation Accuracy: {overall_accuracy:.4f}")

    # Print per-class accuracy.
    for i in range(num_classes):
        if total_per_class[i] > 0:
            class_accuracy = correct_per_class[i] / total_per_class[i]
        else:
            class_accuracy = 0.0
        print(f"Class {i} Accuracy: {class_accuracy:.4f}")


def test(model, test_loader, num_classes=6):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_true = []
    all_preds = []

    progress_bar = tqdm(test_loader, desc="Testing", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            visual_embeds = batch["visual_embeds"].to(device)
            # If your model uses visual_attention_mask or norm_boxes, add them as needed.
            labels = batch["labels"].to(device)

            # Forward pass (adjust if your model's forward expects additional inputs)
            logits = model(input_ids, attention_mask, visual_embeds)
            predictions = torch.argmax(logits, dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_true.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

            progress_bar.set_postfix({"Acc": total_correct / total_samples})

    overall_accuracy = total_correct / total_samples
    print(f"Test Accuracy: {overall_accuracy:.4f}")

    # Generate classification report
    report = classification_report(
        all_true, all_preds, target_names=[f"Class {i}" for i in range(num_classes)]
    )
    print("Classification Report:\n", report)

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_true, all_preds)
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

    # File paths
    MMHS150K_PATH = "../../data/MMHS150K"
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

    # Create dataset and dataloaders
    train_dataset = MMHS150KDataset(train_ids, data, tokenizer, IMG_DIR, OCR_DIR)
    val_dataset = MMHS150KDataset(val_ids, data, tokenizer, IMG_DIR, OCR_DIR)
    test_dataset = MMHS150KDataset(test_ids, data, tokenizer, IMG_DIR, OCR_DIR)

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
        # pin_memory=True
    )

    # Load the checkpoint (adjust the checkpoint_path accordingly)
    # checkpoint_path = "v1/vbert_v1_epoch1_iter2000.pt"
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint)

    # model.eval()  # set the model to evaluation mode
    # validate(model, val_loader)

    model.train()
    train(model, epochs=1)
