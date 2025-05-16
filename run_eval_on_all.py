import os
import torch

from dataset.datamodule import MMHSDataModule
from models.clip_classifier import CLIPClassifier
from core.criteria import SoftFocalLoss
from core.eval import evaluate
from core.io import load_model


def find_checkpoints(ckpt_dir, model_name, version=None):
    model_dir = os.path.join(ckpt_dir, model_name)
    if not os.path.exists(model_dir):
        print(f"No such directory: {model_dir}")
        return []
    files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if version:
        files = [f for f in files if f.startswith(version)]
    files = sorted(files, key=lambda x: int(x.split("epoch")[-1].replace(".pt", "")))
    return [os.path.join(model_dir, f) for f in files]


def evaluate_all_checkpoints(
    data_root: str,
    ckpt_dir: str = "ckpt",
    model_name: str = "CLIPClassifier",
    version: str = "v1",
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    soft_labels: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixes tokenizers warning when num_workers > 0
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm = MMHSDataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    dm.setup()
    val_loader = dm.val_dataloader

    num_classes = 6
    criterion = SoftFocalLoss()

    ckpt_paths = find_checkpoints(ckpt_dir, model_name, version)
    if not ckpt_paths:
        print("No checkpoints found!")
        return

    print(f"Found {len(ckpt_paths)} checkpoints.")
    results = []

    for ckpt_path in ckpt_paths:
        model = CLIPClassifier(num_classes=num_classes, model_name=clip_model_name).to(
            device
        )
        model, _, epoch = load_model(
            ckpt_path, model, optimizer=None, map_location=device
        )
        print(f"\nEvaluating checkpoint: {ckpt_path} (epoch={epoch})")
        avg_loss, loose_acc, strict_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            soft_labels=soft_labels,
            process_batch=dm.process_batch,
        )
        results.append(
            {
                "filename": os.path.basename(ckpt_path),
                "epoch": epoch,
                "loss": avg_loss,
                "loose_acc": loose_acc,
                "strict_acc": strict_acc,
            }
        )
        print(
            f"[{os.path.basename(ckpt_path)}] Epoch {epoch}: "
            f"Val Loss: {avg_loss:.4f}, Loose Acc: {loose_acc:.4f}, Strict Acc: {strict_acc:.4f}"
        )

    # Print summary table
    print("\n\n========= SUMMARY OF ALL CHECKPOINTS =========")
    print(
        f"{'Checkpoint':<30} {'Epoch':<5} {'Loss':<10} {'Loose Acc':<10} {'Strict Acc':<10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['filename']:<30} {r['epoch']:<5} {r['loss']:<10.4f} {r['loose_acc']:<10.4f} {r['strict_acc']:<10.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    evaluate_all_checkpoints(
        data_root="./data/MMHS150K/",
        ckpt_dir="ckpt",
        model_name="CLIPClassifierSoftLabels",
        version="v1",
        batch_size=16,
        num_workers=16,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
        clip_model_name="openai/clip-vit-base-patch32",
        soft_labels=True,
    )
