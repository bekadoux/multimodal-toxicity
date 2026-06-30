import json
from pathlib import Path

import pytest

import scripts.build_aggregated_dataset as aggregation


def test_prepare_output_root_rejects_input_dataset_roots(tmp_path: Path) -> None:
    hateful_root = tmp_path / "hateful_memes"
    pridemm_root = tmp_path / "PrideMM"
    hateful_root.mkdir()
    pridemm_root.mkdir()

    with pytest.raises(ValueError, match="input dataset root"):
        aggregation.prepare_output_root(
            hateful_root,
            hateful_root=hateful_root,
            pridemm_root=pridemm_root,
            overwrite=True,
        )

    with pytest.raises(ValueError, match="input dataset root"):
        aggregation.prepare_output_root(
            pridemm_root,
            hateful_root=hateful_root,
            pridemm_root=pridemm_root,
            overwrite=True,
        )


def test_prepare_output_root_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    hateful_root = tmp_path / "hateful_memes"
    pridemm_root = tmp_path / "PrideMM"
    output_root = tmp_path / "aggregated"
    hateful_root.mkdir()
    pridemm_root.mkdir()
    output_root.mkdir()

    with pytest.raises(ValueError, match="pass --overwrite"):
        aggregation.prepare_output_root(
            output_root,
            hateful_root=hateful_root,
            pridemm_root=pridemm_root,
            overwrite=False,
        )
    assert output_root.exists()


def test_prepare_output_root_rejects_source_parent_before_destructive_delete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "data"
    hateful_root = output_root / "hateful_memes"
    pridemm_root = output_root / "PrideMM"
    hateful_root.mkdir(parents=True)
    pridemm_root.mkdir()

    def fail_if_called(path: Path) -> None:
        pytest.fail(f"rmtree should not be called for dangerous output root: {path}")

    monkeypatch.setattr(aggregation.shutil, "rmtree", fail_if_called)
    with pytest.raises(ValueError):
        aggregation.prepare_output_root(
            output_root,
            hateful_root=hateful_root,
            pridemm_root=pridemm_root,
            overwrite=True,
        )


def test_validate_caption_output_rejects_existing_caption_file(
    tmp_path: Path,
) -> None:
    caption_path = tmp_path / "aggregated_captions.json"
    caption_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Caption file already exists"):
        aggregation.validate_caption_output(
            caption_path,
            {"source_files_found": 1},
            overwrite=False,
        )


def test_write_caption_json_replaces_file_and_removes_temp_path(tmp_path: Path) -> None:
    caption_path = tmp_path / "captions.json"
    caption_path.write_text('{"old": "caption"}\n', encoding="utf-8")

    aggregation.write_caption_json(
        caption_path,
        {"b": "caption b", "a": "caption a"},
        overwrite=True,
    )

    assert json.loads(caption_path.read_text(encoding="utf-8")) == {
        "a": "caption a",
        "b": "caption b",
    }
    assert not caption_path.with_suffix(".json.tmp").exists()
