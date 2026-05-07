from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.switch_backend("Agg")


DEFAULT_DATA_ROOT = Path("data/hateful_memes")
DEFAULT_OUTPUT_PATH = Path("reports/hateful_memes_overlaps.png")
SPLITS = ("train", "dev_seen", "dev_unseen", "test_seen", "test_unseen")
ALLOWED_OVERLAP = frozenset(("dev_seen", "dev_unseen"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Hateful Memes split overlap Euler diagram by sample ID."
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Path to the Hateful Memes dataset root.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="PNG output path.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Missing Hateful Memes split file: {path}")

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


def build_value_sets(
    records_by_split: dict[str, list[dict[str, Any]]],
    field: str,
) -> dict[str, set[str]]:
    if field == "img":
        return {
            split: {Path(record[field]).name for record in records_by_split[split]}
            for split in SPLITS
        }
    return {
        split: {str(record[field]) for record in records_by_split[split]}
        for split in SPLITS
    }


def pairwise_overlaps(value_sets: dict[str, set[str]]) -> dict[frozenset[str], int]:
    overlaps = {}
    for left, right in combinations(SPLITS, 2):
        overlaps[frozenset((left, right))] = len(value_sets[left] & value_sets[right])
    return overlaps


def validate_expected_overlap_structure(
    id_sets: dict[str, set[str]],
    image_sets: dict[str, set[str]],
) -> dict[frozenset[str], int]:
    id_overlaps = pairwise_overlaps(id_sets)
    image_overlaps = pairwise_overlaps(image_sets)
    unexpected = {
        tuple(sorted(pair)): count
        for pair, count in id_overlaps.items()
        if pair != ALLOWED_OVERLAP and count != 0
    }
    if unexpected:
        raise ValueError(
            "Unexpected Hateful Memes ID overlap structure; refusing to draw a "
            f"misleading fixed Euler layout: {unexpected}"
        )

    if image_overlaps != id_overlaps:
        print(
            "Warning: image filename overlaps differ from ID overlaps. "
            "The diagram is drawn by ID overlap only."
        )

    return id_overlaps


def add_circle(
    ax: plt.Axes,
    *,
    xy: tuple[float, float],
    radius: float,
    color: str,
    label: str,
    total: int,
    label_xy: tuple[float, float],
    count_xy: tuple[float, float],
) -> None:
    ax.add_patch(
        Circle(
            xy,
            radius=radius,
            facecolor=color,
            edgecolor=color,
            alpha=0.22,
            linewidth=2.5,
        )
    )
    ax.text(
        *label_xy,
        label,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        *count_xy,
        f"n={total}",
        ha="center",
        va="center",
        fontsize=11,
    )


def draw_euler_diagram(
    ax: plt.Axes,
    id_sets: dict[str, set[str]],
    id_overlaps: dict[frozenset[str], int],
) -> None:
    dev_overlap = id_overlaps[ALLOWED_OVERLAP]
    dev_seen_only = len(id_sets["dev_seen"] - id_sets["dev_unseen"])
    dev_unseen_only = len(id_sets["dev_unseen"] - id_sets["dev_seen"])

    add_circle(
        ax,
        xy=(-3.0, 1.05),
        radius=1.05,
        color="#4c78a8",
        label="train",
        total=len(id_sets["train"]),
        label_xy=(-3.0, 1.2),
        count_xy=(-3.0, 0.82),
    )
    add_circle(
        ax,
        xy=(3.0, 1.05),
        radius=1.0,
        color="#b279a2",
        label="test_seen",
        total=len(id_sets["test_seen"]),
        label_xy=(3.0, 1.2),
        count_xy=(3.0, 0.82),
    )
    add_circle(
        ax,
        xy=(0.0, -1.35),
        radius=1.0,
        color="#e45756",
        label="test_unseen",
        total=len(id_sets["test_unseen"]),
        label_xy=(0.0, -1.2),
        count_xy=(0.0, -1.58),
    )

    ax.add_patch(
        Circle(
            (-0.52, 1.0),
            radius=1.18,
            facecolor="#f58518",
            edgecolor="#f58518",
            alpha=0.25,
            linewidth=2.5,
        )
    )
    ax.add_patch(
        Circle(
            (0.52, 1.0),
            radius=1.18,
            facecolor="#54a24b",
            edgecolor="#54a24b",
            alpha=0.25,
            linewidth=2.5,
        )
    )

    ax.text(-1, 1.52, "dev_seen", ha="center", fontsize=13, fontweight="bold")
    ax.text(1.05, 1.52, "dev_unseen", ha="center", fontsize=13, fontweight="bold")
    ax.text(-1.18, 0.72, f"only\n{dev_seen_only}", ha="center", va="center")
    ax.text(1.18, 0.72, f"only\n{dev_unseen_only}", ha="center", va="center")
    ax.text(
        0.0,
        1.0,
        f"overlap\n{dev_overlap}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        -1,
        1.35,
        f"n={len(id_sets['dev_seen'])}",
        ha="center",
        fontsize=10,
    )
    ax.text(
        1.05,
        1.35,
        f"n={len(id_sets['dev_unseen'])}",
        ha="center",
        fontsize=10,
    )

    ax.set_xlim(-4.35, 4.35)
    ax.set_ylim(-2.45, 2.32)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    output_path = Path(args.output)

    records_by_split = {
        split: load_jsonl(data_root / f"{split}.jsonl") for split in SPLITS
    }
    id_sets = build_value_sets(records_by_split, "id")
    image_sets = build_value_sets(records_by_split, "img")
    id_overlaps = validate_expected_overlap_structure(id_sets, image_sets)

    figure, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    draw_euler_diagram(ax, id_sets, id_overlaps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)

    print(f"Wrote Hateful Memes overlap report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
