from pathlib import Path
from typing import Any

from core.captions import dataset_caption_file
from dataset.datamodule import AggregatedDataModule, HatefulMemesDataModule


def _dataset_name(data_module: Any, data_root: str) -> str:
    if isinstance(data_module, AggregatedDataModule):
        return "aggregated"
    if isinstance(data_module, HatefulMemesDataModule):
        return "hateful_memes"
    return Path(data_root).resolve().name.lower().replace(" ", "_")


def build_checkpoint_metadata(
    data_root: str,
    data_module: Any,
    load_captions: bool,
    source: str | None = None,
) -> dict[str, Any]:
    dataset = _dataset_name(data_module, data_root)
    data_source = source
    if data_source is None:
        data_source = "all" if dataset == "aggregated" else dataset

    captions_file = dataset_caption_file(data_root)
    captions_used = load_captions and captions_file.exists()

    return {
        "dataset": dataset,
        "data_root": str(Path(data_root)),
        "data_root_resolved": str(Path(data_root).resolve()),
        "data_source": data_source,
        "source_filter": source,
        "captions_requested": load_captions,
        "captions_used": captions_used,
        "captions_file": str(captions_file) if captions_used else None,
    }
