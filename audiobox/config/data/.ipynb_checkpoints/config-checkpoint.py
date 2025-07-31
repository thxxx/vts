from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    paths: list[Path] = field(default_factory=list)
    max_audio_len: int = 125
    max_txt_len: int = 128
