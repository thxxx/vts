from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class AudioBoxBaseConfig:
    dim: int = MISSING
    depth: int = MISSING
    heads: int = MISSING
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    kernel_size: int = 31
    text_repo_id: str = MISSING


@dataclass
class AudioBoxGigaConfig(AudioBoxBaseConfig):
    dim: int = 2048
    depth: int = 48
    heads: int = 32
    text_repo_id: str = "google/flan-t5-xl"


@dataclass
class AudioBoxLargeConfig(AudioBoxBaseConfig):
    dim: int = 1024
    depth: int = 24
    heads: int = 16
    text_repo_id: str = "google/flan-t5-base"


@dataclass
class AudioBoxMediumConfig(AudioBoxBaseConfig):
    dim: int = 512
    depth: int = 12
    heads: int = 8
    text_repo_id: str = "google/flan-t5-base"


@dataclass
class AudioBoxSmallConfig(AudioBoxBaseConfig):
    dim: int = 256
    depth: int = 6
    heads: int = 4
    text_repo_id: str = "google/flan-t5-base"
