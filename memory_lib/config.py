from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MemoryConfig:
    base_dir: str
    similarity_threshold: float = 0.88
    top_k: int = 3


def get_default_memory_config() -> MemoryConfig:
    config_path = Path(__file__).with_name("default_config.yaml")
    data = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = (yaml.safe_load(handle) or {}).get("memory", {})
    base_dir = Path(__file__).parent / data.get("base_dir", "vector_store")
    return MemoryConfig(
        base_dir=str(base_dir),
        similarity_threshold=float(data.get("similarity_threshold", 0.88)),
        top_k=int(data.get("top_k", 3)),
    )
