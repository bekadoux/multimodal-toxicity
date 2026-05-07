import os

import torch
from torch import nn

from commands.eval_utils import select_eval_dataloader
from core.eval import append_log, evaluate
from core.io import load_model
from core.logs import build_log_path, make_run_timestamp
from dataset.datamodule import build_eval_data_module
from models.clip_classifier import CLIPClassifier


def find_checkpoints(
    ckpt_dir: str, model_name: str, version: str | None = None
) -> list[str]:
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
    num_classes: int = 2,
    batch_size: int = 64,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    load_captions: bool = True,
    clip_model_name: str = "ViT-L-14",
    clip_pretrained: str = "datacomp_xl_s13b_b90k",
    metadata_file: str = "MMHS150K_GT.json",
    eval_split: str = "val",
    source: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    eval_log_path = build_log_path(
        model_name,
        "eval-all",
        timestamp=make_run_timestamp(),
    )
    print(f"Evaluation log: {eval_log_path}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm = build_eval_data_module(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        load_captions=load_captions,
        num_classes=num_classes,
        metadata_filename=metadata_file,
        source=source,
    )
    dm.setup()
    eval_loader, split_label = select_eval_dataloader(dm, eval_split)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    ckpt_paths = find_checkpoints(ckpt_dir, model_name, version)
    if not ckpt_paths:
        print("No checkpoints found!")
        return

    print(f"Found {len(ckpt_paths)} checkpoints.")
    append_log(eval_log_path, f"Found {len(ckpt_paths)} checkpoints.\n")
    results: list[dict[str, float | int | str]] = []

    for ckpt_path in ckpt_paths:
        model = CLIPClassifier(
            num_classes=num_classes,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
        ).to(device)
        model, _, epoch = load_model(
            ckpt_path,
            model,
            optimizer=None,
            map_location=device,
        )
        print(f"\nEvaluating checkpoint: {ckpt_path} (epoch={epoch})")
        append_log(
            eval_log_path, f"\nEvaluating checkpoint: {ckpt_path} (epoch={epoch})\n"
        )
        metrics = evaluate(
            model,
            eval_loader,
            criterion,
            device,
            process_batch=dm.process_batch,
            log_path=eval_log_path,
        )
        results.append(
            {
                "filename": os.path.basename(ckpt_path),
                "epoch": epoch,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "auroc": metrics["auroc"],
            }
        )
        auroc_str = "N/A" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}"
        print(
            f"[{os.path.basename(ckpt_path)}] Epoch {epoch}: "
            f"{split_label} Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"AUROC: {auroc_str}"
        )
        append_log(
            eval_log_path,
            (
                f"[{os.path.basename(ckpt_path)}] Epoch {epoch}: "
                f"{split_label} Loss: {metrics['loss']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {auroc_str}\n"
            ),
        )

    print("\n\n========= SUMMARY OF ALL CHECKPOINTS =========")
    print(
        f"{'Checkpoint':<30} {'Epoch':<5} {'Loss':<10} {'Accuracy':<10} {'AUROC':<10}"
    )
    print("-" * 70)
    append_log(
        eval_log_path,
        (
            "\n\n========= SUMMARY OF ALL CHECKPOINTS =========\n"
            f"{'Checkpoint':<30} {'Epoch':<5} {'Loss':<10} "
            f"{'Accuracy':<10} {'AUROC':<10}\n"
            f"{'-' * 70}\n"
        ),
    )
    for result in results:
        auroc_str = "N/A" if result["auroc"] is None else f"{result['auroc']:.4f}"
        print(
            f"{result['filename']:<30} {result['epoch']:<5} {result['loss']:<10.4f} "
            f"{result['accuracy']:<10.4f} {auroc_str:<10}"
        )
        append_log(
            eval_log_path,
            (
                f"{result['filename']:<30} {result['epoch']:<5} "
                f"{result['loss']:<10.4f} {result['accuracy']:<10.4f} "
                f"{auroc_str:<10}\n"
            ),
        )
    print("=" * 70)
    append_log(eval_log_path, f"{'=' * 70}\n")
