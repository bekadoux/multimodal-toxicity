from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = REPO_ROOT / "reports" / "logs"


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_log_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "run"


def build_log_path(
    model_name: str,
    log_kind: str,
    *,
    timestamp: str | None = None,
    log_dir: str | Path = DEFAULT_LOG_DIR,
) -> Path:
    timestamp = timestamp or make_run_timestamp()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_name = sanitize_log_component(model_name)
    log_kind = sanitize_log_component(log_kind)
    return log_dir / f"{timestamp}_{model_name}_{log_kind}.log"
