from pathlib import Path

import pytest

import scripts.build_aggregated_dataset as aggregation
from scripts.build_aggregated_dataset import (
    build_aggregated_captions,
    build_records,
    decontaminate_hateful_test_unseen,
    normalize_text_for_overlap,
    resized_dimensions,
)
from scripts.build_mmhs150k_clean import (
    AGREEMENT_MAJORITY,
    AGREEMENT_UNANIMOUS,
    build_candidate_pools,
    select_for_label,
)


def _source_record(
    *,
    label: int,
    text: str,
    source: str = "hateful_memes",
    source_split: str = "train",
    source_id: str = "source-1",
    source_image_path: Path = Path("source.png"),
) -> dict[str, object]:
    return {
        "label": label,
        "text": text,
        "source": source,
        "source_id": source_id,
        "source_split": source_split,
        "source_image": source_image_path.name,
        "source_image_path": source_image_path,
        "source_labels": {"label": label},
    }


def test_normalize_text_for_overlap_casefolds_and_collapses_whitespace() -> None:
    assert normalize_text_for_overlap("  Hello\nWORLD\t ") == "hello world"


def test_decontaminate_hateful_test_unseen_removes_holdout_text_overlap() -> None:
    overlapping = _source_record(
        label=0,
        text="Repeated Text",
        source_split="test_unseen",
        source_id="overlap",
    )
    unique = _source_record(
        label=1,
        text="Unique Text",
        source_split="test_unseen",
        source_id="unique",
    )
    holdout = _source_record(
        label=1,
        text=" repeated   text ",
        source="pridemm",
        source_split="val",
        source_id="holdout",
    )

    kept, summary = decontaminate_hateful_test_unseen(
        [overlapping, unique],
        holdout_records=[holdout],
    )

    assert kept == [unique]
    assert summary["removed"]["total"] == 1
    assert summary["kept"]["total"] == 1
    assert summary["normalization"] == "casefold, strip, collapse whitespace"


def test_collect_source_records_applies_aggregation_split_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hateful_records = {
        "train": [
            _source_record(
                label=0,
                text="hateful train",
                source_id="hm-train",
                source_split="train",
            )
        ],
        "test_unseen": [
            _source_record(
                label=1,
                text="holdout overlap",
                source_id="hm-test-unseen-overlap",
                source_split="test_unseen",
            ),
            _source_record(
                label=1,
                text="unique train text",
                source_id="hm-test-unseen-kept",
                source_split="test_unseen",
            ),
        ],
        "dev_seen": [
            _source_record(
                label=0,
                text="Holdout   Overlap",
                source_id="hm-dev-seen",
                source_split="dev_seen",
            )
        ],
        "test_seen": [
            _source_record(
                label=1,
                text="hateful test",
                source_id="hm-test-seen",
                source_split="test_seen",
            )
        ],
    }
    pridemm_records = {
        "train": [
            _source_record(
                label=1,
                text="pridemm train",
                source="pridemm",
                source_id="pm-train",
                source_split="train",
            )
        ],
        "val": [
            _source_record(
                label=0,
                text="pridemm val",
                source="pridemm",
                source_id="pm-val",
                source_split="val",
            )
        ],
        "test": [
            _source_record(
                label=1,
                text="pridemm test",
                source="pridemm",
                source_id="pm-test",
                source_split="test",
            )
        ],
    }

    monkeypatch.setattr(
        aggregation,
        "load_hateful_records",
        lambda _root, split: hateful_records[split],
    )
    monkeypatch.setattr(
        aggregation,
        "load_pridemm_records",
        lambda _root, split: pridemm_records[split],
    )

    records_by_split, decontamination = aggregation.collect_source_records(
        Path("hateful-root"),
        Path("pridemm-root"),
    )

    assert [record["source_id"] for record in records_by_split["train"]] == [
        "hm-train",
        "hm-test-unseen-kept",
        "pm-train",
    ]
    assert [record["source_id"] for record in records_by_split["val"]] == [
        "hm-dev-seen",
        "pm-val",
    ]
    assert [record["source_id"] for record in records_by_split["test"]] == [
        "hm-test-seen",
        "pm-test",
    ]
    assert decontamination["removed"]["total"] == 1
    assert decontamination["kept"]["total"] == 1


def test_build_records_assigns_synthetic_paths_and_copy_plan(tmp_path: Path) -> None:
    source_image = tmp_path / "source" / "sample.JPG"
    source_records_by_split = {
        "train": [
            _source_record(
                label=1,
                text="train text",
                source_id="train-1",
                source_image_path=source_image,
            )
        ],
        "val": [],
        "test": [],
    }

    split_records, index_records, copy_plan = build_records(
        source_records_by_split,
        tmp_path / "aggregated",
    )

    assert split_records["train"] == [
        {
            "id": "00000001",
            "img": "img/00000001.jpg",
            "label": 1,
            "text": "train text",
            "source": "hateful_memes",
            "source_id": "train-1",
            "source_split": "train",
        }
    ]
    assert index_records[0]["split"] == "train"
    assert copy_plan == [
        {
            "source_path": source_image,
            "destination_path": tmp_path / "aggregated" / "img/00000001.jpg",
            "source": "hateful_memes",
        }
    ]


def test_build_records_rejects_source_images_without_suffix(tmp_path: Path) -> None:
    source_records_by_split = {
        "train": [_source_record(label=0, text="text", source_image_path=Path("bad"))],
        "val": [],
        "test": [],
    }

    with pytest.raises(ValueError, match="no file suffix"):
        build_records(source_records_by_split, tmp_path / "aggregated")


def test_build_aggregated_captions_remaps_source_keys() -> None:
    index_records = [
        {
            "source": "hateful_memes",
            "source_image": "img/a.png",
            "img": "img/00000001.png",
        },
        {
            "source": "pridemm",
            "source_image": "Images/b.png",
            "img": "img/00000002.png",
        },
    ]
    captions, summary = build_aggregated_captions(
        index_records,
        {
            "hateful_memes": {"img/a.png": "caption a"},
            "pridemm": {},
        },
        {
            "hateful_memes": Path("captions/hateful_memes_captions.json"),
            "pridemm": Path("captions/pridemm_captions.json"),
        },
    )

    assert captions == {"img/00000001.png": "caption a"}
    assert summary["matched"] == 1
    assert summary["missing"] == 1
    assert summary["by_source"]["hateful_memes"]["matched"] == 1
    assert summary["by_source"]["pridemm"]["missing"] == 1


def test_resized_dimensions_preserves_small_or_disabled_images() -> None:
    assert resized_dimensions(100, 100, max_pixels=0) == (100, 100)
    assert resized_dimensions(100, 100, max_pixels=10_000) == (100, 100)
    assert resized_dimensions(100, 100, max_pixels=2_500) == (50, 50)


def test_mmhs_candidate_pool_policy_groups_and_excludes_samples() -> None:
    metadata = {
        "unanimous": {"labels": [1, 1, 1]},
        "majority": {"labels": [2, 2, 0]},
        "religion": {"labels": [4, 4, 4]},
        "no_agreement": {"labels": [1, 2, 3]},
        "bad_count": {"labels": [1, 1]},
    }

    pools, stats = build_candidate_pools(metadata)

    assert [tweet_id for tweet_id, _ in pools[1][AGREEMENT_UNANIMOUS]] == ["unanimous"]
    assert [tweet_id for tweet_id, _ in pools[2][AGREEMENT_MAJORITY]] == ["majority"]
    assert stats["excluded_religion_vote"] == 1
    assert stats["excluded_no_agreement"] == 1
    assert stats["excluded_non_3_labels"] == 1


def test_mmhs_select_for_label_is_deterministic() -> None:
    pools = {
        1: {
            AGREEMENT_UNANIMOUS: [
                ("u1", {"labels": [1, 1, 1]}),
                ("u2", {"labels": [1, 1, 1]}),
            ],
            AGREEMENT_MAJORITY: [
                ("m1", {"labels": [1, 1, 0]}),
                ("m2", {"labels": [1, 1, 0]}),
            ],
        }
    }

    first = select_for_label(1, pools, target_count=2, base_seed=123)
    second = select_for_label(1, pools, target_count=2, base_seed=123)

    assert first == second
    assert len(first[AGREEMENT_UNANIMOUS]) == 1
    assert len(first[AGREEMENT_MAJORITY]) == 1
