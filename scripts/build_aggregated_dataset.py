from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HATEFUL_ROOT = Path("data/hateful_memes")
DEFAULT_PRIDEMM_ROOT = Path("data/PrideMM")
DEFAULT_OUTPUT_ROOT = Path("data/aggregated")
CAPTIONS_DIR = REPO_ROOT / "captions"
SPLITS = ("train", "val", "test")
SOURCES = ("hateful_memes", "pridemm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Hateful-Memes-style aggregated binary hate/non-hate dataset "
            "from Hateful Memes and PrideMM."
        )
    )
    parser.add_argument(
        "--hateful-root",
        default=str(DEFAULT_HATEFUL_ROOT),
        help="Path to the Hateful Memes dataset root.",
    )
    parser.add_argument(
        "--pridemm-root",
        default=str(DEFAULT_PRIDEMM_ROOT),
        help="Path to the PrideMM dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Path where the aggregated dataset should be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-root if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned summary without writing files.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Missing JSONL file: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing CSV file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dataset_caption_file(data_root: str | Path, suffix: str = ".json") -> Path:
    dataset_name = Path(data_root).resolve().name.lower().replace(" ", "_")
    return CAPTIONS_DIR / f"{dataset_name}_captions{suffix}"


def load_caption_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        captions = json.load(f)

    if not isinstance(captions, dict):
        raise ValueError(f"Caption file must contain a JSON object: {path}")

    normalized_captions = {}
    for key, value in captions.items():
        if not isinstance(value, str):
            raise ValueError(f"Caption value for {key!r} is not a string in {path}")
        caption = value.strip()
        if caption:
            normalized_captions[str(key)] = caption
    return normalized_captions


def parse_binary_label(value: object, *, context: str) -> int:
    try:
        label = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid binary label for {context}: {value!r}") from exc

    if label not in (0, 1):
        raise ValueError(f"Expected binary label 0 or 1 for {context}, got {label}")
    return label


def require_text(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Missing text for {context}")
    return value


def normalize_text_for_overlap(text: str) -> str:
    return " ".join(text.casefold().split())


def load_hateful_records(
    data_root: Path,
    source_split: str,
) -> list[dict[str, Any]]:
    jsonl_path = data_root / f"{source_split}.jsonl"
    image_root = data_root / "img"
    records = []

    for entry in load_jsonl(jsonl_path):
        source_id = str(entry.get("id", ""))
        if not source_id:
            raise ValueError(f"Missing Hateful Memes id in {jsonl_path}")

        source_image = require_text(
            entry.get("img"),
            context=f"Hateful Memes {source_split} sample {source_id}",
        )
        source_image_path = data_root / source_image
        if not source_image_path.exists():
            fallback_path = image_root / Path(source_image).name
            if fallback_path.exists():
                source_image_path = fallback_path
            else:
                raise ValueError(f"Missing Hateful Memes image: {source_image_path}")

        label = parse_binary_label(
            entry.get("label"),
            context=f"Hateful Memes {source_split} sample {source_id}",
        )
        text = require_text(
            entry.get("text"),
            context=f"Hateful Memes {source_split} sample {source_id}",
        )

        records.append(
            {
                "label": label,
                "text": text,
                "source": "hateful_memes",
                "source_id": source_id,
                "source_split": source_split,
                "source_image": source_image_path.relative_to(data_root).as_posix(),
                "source_image_path": source_image_path,
                "source_labels": {
                    "label": label,
                },
            }
        )

    return records


def load_pridemm_records(
    data_root: Path,
    source_split: str,
) -> list[dict[str, Any]]:
    csv_path = data_root / "PrideMM.csv"
    image_root = data_root / "Images"
    records = []

    for row_number, row in enumerate(read_csv(csv_path), start=2):
        if row.get("split") != source_split:
            continue

        image_name = require_text(
            row.get("name"),
            context=f"PrideMM row {row_number}",
        )
        source_image_path = image_root / image_name
        if not source_image_path.exists():
            raise ValueError(f"Missing PrideMM image: {source_image_path}")

        source_id = Path(image_name).stem
        label = parse_binary_label(
            row.get("hate"),
            context=f"PrideMM {source_split} sample {source_id}",
        )
        text = require_text(
            row.get("text"),
            context=f"PrideMM {source_split} sample {source_id}",
        )

        records.append(
            {
                "label": label,
                "text": text,
                "source": "pridemm",
                "source_id": source_id,
                "source_split": source_split,
                "source_image": source_image_path.relative_to(data_root).as_posix(),
                "source_image_path": source_image_path,
                "source_labels": {
                    "hate": label,
                    "target": row.get("target"),
                    "stance": row.get("stance"),
                    "humour": row.get("humour"),
                },
            }
        )

    return records


def collect_source_records(
    hateful_root: Path,
    pridemm_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    hateful_train = load_hateful_records(hateful_root, "train")
    hateful_test_unseen = load_hateful_records(hateful_root, "test_unseen")
    pridemm_train = load_pridemm_records(pridemm_root, "train")
    val_records = [
        *load_hateful_records(hateful_root, "dev_seen"),
        *load_pridemm_records(pridemm_root, "val"),
    ]
    test_records = [
        *load_hateful_records(hateful_root, "test_seen"),
        *load_pridemm_records(pridemm_root, "test"),
    ]

    kept_test_unseen, decontamination = decontaminate_hateful_test_unseen(
        hateful_test_unseen,
        holdout_records=[*val_records, *test_records],
    )

    return (
        {
            "train": [
                *hateful_train,
                *kept_test_unseen,
                *pridemm_train,
            ],
            "val": val_records,
            "test": test_records,
        },
        decontamination,
    )


def decontaminate_hateful_test_unseen(
    test_unseen_records: list[dict[str, Any]],
    *,
    holdout_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holdout_text_keys = {
        key
        for record in holdout_records
        if (key := normalize_text_for_overlap(record["text"]))
    }

    kept_records = []
    removed_records = []
    for record in test_unseen_records:
        text_key = normalize_text_for_overlap(record["text"])
        if text_key and text_key in holdout_text_keys:
            removed_records.append(record)
        else:
            kept_records.append(record)

    summary = {
        "policy": (
            "Only Hateful Memes test_unseen records added to the aggregated "
            "training split are removed when normalized text overlaps the "
            "aggregated validation or test split. Other training sources and "
            "all validation/test records are left unchanged."
        ),
        "normalization": "casefold, strip, collapse whitespace",
        "filtered_source_split": "hateful_memes:test_unseen",
        "holdout_splits": ["val", "test"],
        "holdout_unique_texts": len(holdout_text_keys),
        "before": summarize_split(test_unseen_records),
        "kept": summarize_split(kept_records),
        "removed": summarize_split(removed_records),
    }
    return kept_records, summary


def build_records(
    source_records_by_split: dict[str, list[dict[str, Any]]],
    output_root: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[tuple[Path, Path]],
]:
    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    index_records = []
    copy_plan = []
    next_index = 1

    for split in SPLITS:
        for source_record in source_records_by_split[split]:
            sample_id = f"{next_index:08d}"
            source_image_path = source_record["source_image_path"]
            suffix = source_image_path.suffix.lower()
            if not suffix:
                raise ValueError(
                    f"Source image has no file suffix: {source_image_path}"
                )

            image_rel = f"img/{sample_id}{suffix}"
            training_record = {
                "id": sample_id,
                "img": image_rel,
                "label": source_record["label"],
                "text": source_record["text"],
                "source": source_record["source"],
                "source_id": source_record["source_id"],
                "source_split": source_record["source_split"],
            }
            index_record = {
                **training_record,
                "split": split,
                "source_image": source_record["source_image"],
                "source_labels": source_record["source_labels"],
            }

            split_records[split].append(training_record)
            index_records.append(index_record)
            copy_plan.append((source_image_path, output_root / image_rel))
            next_index += 1

    return split_records, index_records, copy_plan


def load_source_captions(
    *,
    hateful_root: Path,
    pridemm_root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Path]]:
    caption_paths = {
        "hateful_memes": dataset_caption_file(hateful_root),
        "pridemm": dataset_caption_file(pridemm_root),
    }
    source_captions = {
        source: load_caption_json(caption_paths[source]) for source in SOURCES
    }
    return source_captions, caption_paths


def build_aggregated_captions(
    index_records: list[dict[str, Any]],
    source_captions: dict[str, dict[str, str]],
    caption_paths: dict[str, Path],
) -> tuple[dict[str, str], dict[str, Any]]:
    aggregate_captions = {}
    by_source = {
        source: {
            "caption_file": str(caption_paths[source]),
            "caption_file_found": caption_paths[source].exists(),
            "source_captions": len(source_captions[source]),
            "records": 0,
            "matched": 0,
            "missing": 0,
        }
        for source in SOURCES
    }

    for record in index_records:
        source = record["source"]
        source_image = record["source_image"]
        by_source[source]["records"] += 1

        caption = source_captions.get(source, {}).get(source_image)
        if caption:
            aggregate_captions[record["img"]] = caption
            by_source[source]["matched"] += 1
        else:
            by_source[source]["missing"] += 1

    return aggregate_captions, {
        "output_path": str(dataset_caption_file(DEFAULT_OUTPUT_ROOT)),
        "total": len(index_records),
        "matched": len(aggregate_captions),
        "missing": len(index_records) - len(aggregate_captions),
        "source_files_found": sum(
            1 for source in SOURCES if by_source[source]["caption_file_found"]
        ),
        "by_source": by_source,
    }


def counter_as_dict(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def nested_counter_as_dict(counter: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {key: counter_as_dict(value) for key, value in sorted(counter.items())}


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(record["label"] for record in records)
    sources = Counter(record["source"] for record in records)
    source_splits = Counter(
        f"{record['source']}:{record['source_split']}" for record in records
    )
    labels_by_source: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        labels_by_source[record["source"]][record["label"]] += 1

    return {
        "total": len(records),
        "by_label": counter_as_dict(labels),
        "by_source": counter_as_dict(sources),
        "by_source_split": counter_as_dict(source_splits),
        "by_label_by_source": nested_counter_as_dict(labels_by_source),
    }


def build_manifest(
    split_records: dict[str, list[dict[str, Any]]],
    *,
    hateful_root: Path,
    pridemm_root: Path,
    decontamination: dict[str, Any],
) -> dict[str, Any]:
    split_summaries = {split: summarize_split(split_records[split]) for split in SPLITS}
    total_records = sum(summary["total"] for summary in split_summaries.values())

    return {
        "total_records": total_records,
        "label_space": {
            "0": "non_hate",
            "1": "hate",
        },
        "source_roots": {
            "hateful_memes": str(hateful_root),
            "pridemm": str(pridemm_root),
        },
        "split_policy": {
            "train": [
                "hateful_memes:train",
                "hateful_memes:test_unseen",
                "pridemm:train",
            ],
            "val": [
                "hateful_memes:dev_seen",
                "pridemm:val",
            ],
            "test": [
                "hateful_memes:test_seen",
                "pridemm:test",
            ],
        },
        "image_naming": (
            "Sequential synthetic filenames are assigned in split/source order. "
            "Original image paths are stored in index.jsonl."
        ),
        "decontamination": decontamination,
        "splits": split_summaries,
    }


def prepare_output_root(
    output_root: Path,
    *,
    hateful_root: Path,
    pridemm_root: Path,
    overwrite: bool,
) -> None:
    resolved_output = output_root.resolve()
    source_roots = {hateful_root.resolve(), pridemm_root.resolve()}
    if resolved_output in source_roots:
        raise ValueError("output-root must not be the same as an input dataset root")

    if output_root.exists():
        if not overwrite:
            raise ValueError(
                f"Output root already exists, pass --overwrite: {output_root}"
            )
        shutil.rmtree(output_root)

    (output_root / "img").mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_caption_json(
    path: Path, captions: dict[str, str], *, overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"Caption file already exists, pass --overwrite: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(captions.items())), f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def should_write_caption_json(summary: dict[str, Any]) -> bool:
    return summary["source_files_found"] > 0


def validate_caption_output(
    path: Path, summary: dict[str, Any], *, overwrite: bool
) -> None:
    if should_write_caption_json(summary) and path.exists() and not overwrite:
        raise ValueError(f"Caption file already exists, pass --overwrite: {path}")


def write_dataset(
    output_root: Path,
    split_records: dict[str, list[dict[str, Any]]],
    index_records: list[dict[str, Any]],
    copy_plan: list[tuple[Path, Path]],
    manifest: dict[str, Any],
) -> None:
    for source_path, destination_path in copy_plan:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    for split in SPLITS:
        write_jsonl(output_root / f"{split}.jsonl", split_records[split])

    write_jsonl(output_root / "index.jsonl", index_records)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_summary(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    dry_run: bool,
    caption_summary: dict[str, Any],
) -> None:
    action = "Dry run complete" if dry_run else "Aggregated dataset written"
    print(f"{action}: {output_root}")
    print(f"Total records: {manifest['total_records']}")
    for split in SPLITS:
        summary = manifest["splits"][split]
        print(
            f"{split}: total={summary['total']} "
            f"labels={summary['by_label']} sources={summary['by_source']}"
        )
    decontamination = manifest["decontamination"]
    removed = decontamination["removed"]
    kept = decontamination["kept"]
    print(
        "decontamination: "
        f"filtered={decontamination['filtered_source_split']} "
        f"removed={removed['total']} kept={kept['total']} "
        f"removed_labels={removed['by_label']} kept_labels={kept['by_label']}"
    )
    print(
        "captions: "
        f"output={caption_summary['output_path']} "
        f"matched={caption_summary['matched']} "
        f"missing={caption_summary['missing']}"
    )
    for source in SOURCES:
        source_summary = caption_summary["by_source"][source]
        status = "found" if source_summary["caption_file_found"] else "missing"
        print(
            f"captions[{source}]: file={status} "
            f"source_captions={source_summary['source_captions']} "
            f"records={source_summary['records']} "
            f"matched={source_summary['matched']} "
            f"missing={source_summary['missing']}"
        )


def main() -> int:
    args = parse_args()
    hateful_root = Path(args.hateful_root)
    pridemm_root = Path(args.pridemm_root)
    output_root = Path(args.output_root)

    source_records_by_split, decontamination = collect_source_records(
        hateful_root,
        pridemm_root,
    )
    split_records, index_records, copy_plan = build_records(
        source_records_by_split,
        output_root,
    )
    source_captions, caption_paths = load_source_captions(
        hateful_root=hateful_root,
        pridemm_root=pridemm_root,
    )
    aggregate_captions, caption_summary = build_aggregated_captions(
        index_records,
        source_captions,
        caption_paths,
    )
    caption_output_path = dataset_caption_file(output_root)
    caption_summary["output_path"] = str(caption_output_path)
    manifest = build_manifest(
        split_records,
        hateful_root=hateful_root,
        pridemm_root=pridemm_root,
        decontamination=decontamination,
    )

    if not args.dry_run:
        validate_caption_output(
            caption_output_path,
            caption_summary,
            overwrite=args.overwrite,
        )
        prepare_output_root(
            output_root,
            hateful_root=hateful_root,
            pridemm_root=pridemm_root,
            overwrite=args.overwrite,
        )
        write_dataset(output_root, split_records, index_records, copy_plan, manifest)
        if should_write_caption_json(caption_summary):
            write_caption_json(
                caption_output_path,
                aggregate_captions,
                overwrite=args.overwrite,
            )

    print_summary(
        manifest,
        output_root=output_root,
        dry_run=args.dry_run,
        caption_summary=caption_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
