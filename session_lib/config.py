from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SessionConfig:
    base_dir: str
    json_indent: int
    verbose: bool
    overwrite_corrupt: bool


def get_default_session_config() -> SessionConfig:
    """Load session defaults from session_lib/default_config.yaml."""
    config_path = Path(__file__).with_name("default_config.yaml")
    if not config_path.exists():
        return _defaults()

    with config_path.open("r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    cfg = yaml_cfg.get("session", {})
    raw_base = cfg.get("base_dir", "session_logs")

    # Resolve relative paths against the session_lib directory itself,
    # matching the pattern used by logging_lib and llm_api_lib.
    base_path = Path(__file__).parent / raw_base

    return SessionConfig(
        base_dir=str(base_path),
        json_indent=int(cfg.get("json_indent", 2)),
        verbose=bool(cfg.get("verbose", True)),
        overwrite_corrupt=bool(cfg.get("overwrite_corrupt", True)),
    )


def _defaults() -> SessionConfig:
    return SessionConfig(
        base_dir=str(Path(__file__).parent / "session_logs"),
        json_indent=2,
        verbose=True,
        overwrite_corrupt=True,
    )
