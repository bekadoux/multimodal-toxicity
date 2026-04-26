from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

CLASS_NAMES = {
    0: "NotHate",
    1: "Racist",
    2: "Sexist",
    3: "Homophobe",
    4: "Religion",
    5: "OtherHate",
}

RELIGION_LABEL = 4
STRICT_ANNOTATOR_COUNT = 3
HATE_LABELS = (1, 2, 3, 5)
NON_HATE_LABEL = 0
TARGET_COUNT_PER_HATE_CLASS = 1000
TARGET_NON_HATE_COUNT = 4000
DEFAULT_SEED = 1337
AGREEMENT_UNANIMOUS = "3/3"
AGREEMENT_MAJORITY = "2/3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cleaned MMHS150K evaluation subset with fixed class "
            "and annotator-agreement quotas."
        )
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        default="data/MMHS150K",
        help="Path to the MMHS150K dataset root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed used for deterministic sampling.",
    )
    return parser.parse_args()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def load_metadata(json_path: Path) -> dict[str, dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def agreement_for_labels(labels: list[int]) -> tuple[int | None, str | None]:
    counts = Counter(labels)
    majority_label, majority_count = counts.most_common(1)[0]
    if majority_count == STRICT_ANNOTATOR_COUNT:
        return majority_label, AGREEMENT_UNANIMOUS
    if majority_count == STRICT_ANNOTATOR_COUNT - 1:
        return majority_label, AGREEMENT_MAJORITY
    return None, None


def build_candidate_pools(
    metadata: dict[str, dict[str, Any]],
) -> tuple[dict[int, dict[str, list[tuple[str, dict[str, Any]]]]], Counter]:
    pools: dict[int, dict[str, list[tuple[str, dict[str, Any]]]]] = defaultdict(
        lambda: {
            AGREEMENT_UNANIMOUS: [],
            AGREEMENT_MAJORITY: [],
        }
    )
    stats: Counter = Counter()

    for tweet_id, sample in metadata.items():
        stats["total_samples"] += 1

        labels = sample.get("labels", [])
        if len(labels) != STRICT_ANNOTATOR_COUNT:
            stats["excluded_non_3_labels"] += 1
            continue

        if RELIGION_LABEL in labels:
            stats["excluded_religion_vote"] += 1
            continue

        majority_label, agreement = agreement_for_labels(labels)
        if majority_label is None or agreement is None:
            stats["excluded_no_agreement"] += 1
            continue

        if majority_label not in CLASS_NAMES or majority_label == RELIGION_LABEL:
            stats["excluded_unexpected_majority_label"] += 1
            continue

        pools[majority_label][agreement].append((tweet_id, sample))
        stats[f"candidate_{majority_label}_{agreement}"] += 1

    return pools, stats


def sample_entries(
    entries: list[tuple[str, dict[str, Any]]],
    count: int,
    *,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    if count == 0:
        return []
    ordered_entries = sorted(entries, key=lambda item: item[0])
    rng = random.Random(seed)
    return sorted(rng.sample(ordered_entries, count), key=lambda item: item[0])


def select_for_label(
    label: int,
    pools: dict[int, dict[str, list[tuple[str, dict[str, Any]]]]],
    *,
    target_count: int,
    base_seed: int,
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    unanimous_entries = pools[label][AGREEMENT_UNANIMOUS]
    majority_entries = pools[label][AGREEMENT_MAJORITY]

    unanimous_count = min(target_count // 2, len(unanimous_entries))
    majority_count = target_count - unanimous_count

    if len(majority_entries) < majority_count:
        raise ValueError(
            f"Not enough {AGREEMENT_MAJORITY} samples for {CLASS_NAMES[label]}: "
            f"need {majority_count}, found {len(majority_entries)}"
        )

    return {
        AGREEMENT_UNANIMOUS: sample_entries(
            unanimous_entries,
            unanimous_count,
            seed=stable_seed(base_seed, label, AGREEMENT_UNANIMOUS),
        ),
        AGREEMENT_MAJORITY: sample_entries(
            majority_entries,
            majority_count,
            seed=stable_seed(base_seed, label, AGREEMENT_MAJORITY),
        ),
    }


def enrich_sample(
    sample: dict[str, Any],
    *,
    majority_label: int,
    agreement: str,
) -> dict[str, Any]:
    enriched = dict(sample)
    enriched["majority_label"] = majority_label
    enriched["majority_label_str"] = CLASS_NAMES[majority_label]
    enriched["agreement"] = agreement
    return enriched


def build_output_records(
    selected: dict[int, dict[str, list[tuple[str, dict[str, Any]]]]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for label in (NON_HATE_LABEL, *HATE_LABELS):
        for agreement in (AGREEMENT_UNANIMOUS, AGREEMENT_MAJORITY):
            for tweet_id, sample in selected[label][agreement]:
                records[tweet_id] = enrich_sample(
                    sample,
                    majority_label=label,
                    agreement=agreement,
                )
    return dict(sorted(records.items(), key=lambda item: item[0]))


def print_summary(
    stats: Counter,
    selected: dict[int, dict[str, list[tuple[str, dict[str, Any]]]]],
    output_path: Path,
    seed: int,
) -> None:
    print(f"Seed: {seed}")
    print(f"Total samples scanned: {stats['total_samples']}")
    print(
        "Excluded samples: "
        f"non-3-label={stats['excluded_non_3_labels']}, "
        f"religion-vote={stats['excluded_religion_vote']}, "
        f"no-agreement={stats['excluded_no_agreement']}"
    )
    print("Selection summary:")

    for label in (NON_HATE_LABEL, *HATE_LABELS):
        selected_unanimous = len(selected[label][AGREEMENT_UNANIMOUS])
        selected_majority = len(selected[label][AGREEMENT_MAJORITY])
        available_unanimous = stats[f"candidate_{label}_{AGREEMENT_UNANIMOUS}"]
        available_majority = stats[f"candidate_{label}_{AGREEMENT_MAJORITY}"]
        print(
            f"  {label} {CLASS_NAMES[label]:<10} "
            f"selected={selected_unanimous + selected_majority:>4} "
            f"({AGREEMENT_UNANIMOUS}={selected_unanimous:>4}, "
            f"{AGREEMENT_MAJORITY}={selected_majority:>4}) "
            f"available=({AGREEMENT_UNANIMOUS}={available_unanimous:>5}, "
            f"{AGREEMENT_MAJORITY}={available_majority:>5})"
        )

    total_selected = sum(
        len(selected[label][AGREEMENT_UNANIMOUS])
        + len(selected[label][AGREEMENT_MAJORITY])
        for label in (NON_HATE_LABEL, *HATE_LABELS)
    )
    print(f"Wrote {total_selected} selected samples to {output_path}")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    input_path = data_root / "MMHS150K_GT.json"
    output_path = data_root / "MMHS150K_clean.json"

    if not input_path.exists():
        raise ValueError(f"Missing MMHS150K metadata file: {input_path}")

    metadata = load_metadata(input_path)
    pools, stats = build_candidate_pools(metadata)

    selected = {
        NON_HATE_LABEL: select_for_label(
            NON_HATE_LABEL,
            pools,
            target_count=TARGET_NON_HATE_COUNT,
            base_seed=args.seed,
        )
    }
    for label in HATE_LABELS:
        selected[label] = select_for_label(
            label,
            pools,
            target_count=TARGET_COUNT_PER_HATE_CLASS,
            base_seed=args.seed,
        )

    output_records = build_output_records(selected)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)

    print_summary(stats, selected, output_path, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
