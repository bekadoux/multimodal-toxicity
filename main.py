import argparse
import os

from core.captions import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)

# Keep the default conservative because train/val loaders can both keep worker
# pools alive during long runs when persistent workers are enabled.
DEFAULT_NUM_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))


def add_shared_dataloader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--captions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load image caption JSON if available.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal toxicity CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_subparsers = train_parser.add_subparsers(dest="model", required=True)

    train_clip_parser = train_subparsers.add_parser("clip", help="Train CLIP")
    train_clip_parser.add_argument(
        "data_root", nargs="?", default="./data/hateful_memes/"
    )
    train_clip_parser.add_argument("--model-name", default="CLIPClassifier")
    train_clip_parser.add_argument("--version", default="v1")
    train_clip_parser.add_argument("--num-classes", type=int, default=2)
    train_clip_parser.add_argument("--batch-size", type=int, default=16)
    train_clip_parser.add_argument("--max-epochs", type=int, default=200)
    train_clip_parser.add_argument("--patience", type=int, default=15)
    train_clip_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_clip_parser.add_argument("--checkpoint-limit", type=int, default=3)
    train_clip_parser.add_argument("--lr", type=float, default=1e-5)
    train_clip_parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS
    )
    train_clip_parser.add_argument("--prefetch-factor", type=int, default=8)
    train_clip_parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_clip_parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_clip_parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
    )
    train_clip_parser.add_argument(
        "--captions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load image caption JSON if available.",
    )

    train_vbert_parser = train_subparsers.add_parser(
        "vbert",
        help="Train VisualBERT",
    )
    train_vbert_parser.add_argument(
        "data_root", nargs="?", default="./data/hateful_memes/"
    )
    train_vbert_parser.add_argument("--model-name", default="VisualBERT")
    train_vbert_parser.add_argument("--version", default="v1")
    train_vbert_parser.add_argument("--num-classes", type=int, default=2)
    train_vbert_parser.add_argument("--batch-size", type=int, default=16)
    train_vbert_parser.add_argument("--max-epochs", type=int, default=200)
    train_vbert_parser.add_argument("--patience", type=int, default=15)
    train_vbert_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_vbert_parser.add_argument("--checkpoint-limit", type=int, default=3)
    train_vbert_parser.add_argument("--lr", type=float, default=2e-5)
    train_vbert_parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS
    )
    train_vbert_parser.add_argument("--prefetch-factor", type=int, default=8)
    train_vbert_parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_vbert_parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_vbert_parser.add_argument(
        "--captions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load image caption JSON if available.",
    )
    train_vbert_parser.add_argument("--max-visual-tokens", type=int, default=16)

    eval_parser = subparsers.add_parser("eval", help="Run single-checkpoint evaluation")
    eval_subparsers = eval_parser.add_subparsers(dest="model", required=True)

    eval_clip_parser = eval_subparsers.add_parser("clip", help="Evaluate CLIP")
    eval_clip_parser.add_argument("checkpoint_path")
    eval_clip_parser.add_argument("data_root", nargs="?", default="data/hateful_memes/")
    eval_clip_parser.add_argument("--num-classes", type=int, default=2)
    add_shared_dataloader_args(eval_clip_parser)
    eval_clip_parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
    )

    eval_vbert_parser = eval_subparsers.add_parser("vbert", help="Evaluate VisualBERT")
    eval_vbert_parser.add_argument("checkpoint_path")
    eval_vbert_parser.add_argument(
        "data_root", nargs="?", default="data/hateful_memes/"
    )
    eval_vbert_parser.add_argument("--num-classes", type=int, default=2)
    add_shared_dataloader_args(eval_vbert_parser)
    eval_vbert_parser.add_argument("--max-visual-tokens", type=int, default=16)

    eval_all_parser = subparsers.add_parser(
        "eval-all",
        help="Evaluate all matching CLIP checkpoints",
    )
    eval_all_parser.add_argument(
        "data_root", nargs="?", default="./data/hateful_memes/"
    )
    eval_all_parser.add_argument("--ckpt-dir", default="ckpt")
    eval_all_parser.add_argument("--model-name", default="CLIPClassifier")
    eval_all_parser.add_argument("--version", default="v1")
    eval_all_parser.add_argument("--num-classes", type=int, default=2)
    eval_all_parser.add_argument("--batch-size", type=int, default=16)
    eval_all_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    eval_all_parser.add_argument("--prefetch-factor", type=int, default=8)
    eval_all_parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eval_all_parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eval_all_parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
    )
    eval_all_parser.add_argument(
        "--captions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load image caption JSON if available.",
    )

    caption_parser = subparsers.add_parser(
        "caption",
        help="Generate image captions",
    )
    caption_parser.add_argument(
        "data_root",
        help="Dataset root used to scan images and derive the output caption path.",
    )
    caption_parser.add_argument(
        "--image-dir",
        default=None,
        help=(
            "Optional image directory under data_root. Defaults to scanning "
            "data_root recursively."
        ),
    )
    caption_parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080",
        help="Base llama.cpp server URL or full /v1/chat/completions endpoint.",
    )
    caption_parser.add_argument("--prompt", default=None)
    caption_parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    caption_parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    caption_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    caption_parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    caption_parser.add_argument("--seed", type=int, default=42)
    caption_parser.add_argument("--timeout", type=float, default=120.0)
    caption_parser.add_argument("--retries", type=int, default=3)
    caption_parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    caption_parser.add_argument("--max-images", type=int, default=None)
    caption_parser.add_argument(
        "--output",
        default=None,
        help="Optional override for the final caption JSON path.",
    )
    caption_parser.add_argument(
        "--debug-responses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write full llama.cpp JSON responses to a debug JSONL file.",
    )
    caption_parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request thinking-enabled captions when the server allows it.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train" and args.model == "clip":
        from commands.train_clip import train_clip

        train_clip(
            data_root=args.data_root,
            model_name=args.model_name,
            version=args.version,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            checkpoint_limit=args.checkpoint_limit,
            lr=args.lr,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            load_captions=args.captions,
            clip_model_name=args.clip_model_name,
        )
        return

    if args.command == "train" and args.model == "vbert":
        from commands.train_vbert import train_vbert

        train_vbert(
            data_root=args.data_root,
            model_name=args.model_name,
            version=args.version,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            checkpoint_limit=args.checkpoint_limit,
            lr=args.lr,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            load_captions=args.captions,
            max_visual_tokens=args.max_visual_tokens,
        )
        return

    if args.command == "eval" and args.model == "clip":
        from commands.eval_clip import validate_clip

        validate_clip(
            checkpoint_path=args.checkpoint_path,
            data_root=args.data_root,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            load_captions=args.captions,
            clip_model_name=args.clip_model_name,
        )
        return

    if args.command == "eval" and args.model == "vbert":
        from commands.eval_vbert import validate_vbert

        validate_vbert(
            checkpoint_path=args.checkpoint_path,
            data_root=args.data_root,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            load_captions=args.captions,
            max_visual_tokens=args.max_visual_tokens,
        )
        return

    if args.command == "eval-all":
        from commands.eval_all import evaluate_all_checkpoints

        evaluate_all_checkpoints(
            data_root=args.data_root,
            ckpt_dir=args.ckpt_dir,
            model_name=args.model_name,
            version=args.version,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            load_captions=args.captions,
            clip_model_name=args.clip_model_name,
        )
        return

    if args.command == "caption":
        from commands.generate_captions import generate_image_captions

        generate_image_captions(
            data_root=args.data_root,
            image_dir=args.image_dir,
            server_url=args.server_url,
            prompt=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
            max_images=args.max_images,
            output_path=args.output,
            debug_responses=args.debug_responses,
            reasoning=args.reasoning,
        )
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
