from dataclasses import dataclass, field

from omegaconf import MISSING, DictConfig

from config.data.config import DataConfig
from config.model.config import AudioBoxBaseConfig
from config.train.config import TrainConfig

defaults = ["_self_", {"audiobox": "large"}, {"data": "base"}, {"train": "base"}]


@dataclass
class Config:
    defaults: list = field(default_factory=lambda: defaults)

    audiobox: AudioBoxBaseConfig = MISSING
    data: DataConfig = MISSING
    train: TrainConfig = MISSING

    voco: str = "mel"


def dict_to_config(cfg: dict | DictConfig):
    audiobox_config = AudioBoxBaseConfig(**cfg["audiobox"])
    data_config = DataConfig(**cfg["data"])
    train_config = TrainConfig(**cfg["train"])
    return Config(
        audiobox=audiobox_config,
        data=data_config,
        train=train_config,
        voco=cfg["voco"],
    )
