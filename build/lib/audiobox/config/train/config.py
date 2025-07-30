from dataclasses import dataclass


@dataclass
class TrainConfig:
    acc: int = 1
    batch_size: int = 4
    base_path: str = ""
    compile: bool = False
    early_stop: bool = False
    fast_dev_run: bool = False
    gradient_clip_val: float = 0.2
    lr: float = 2e-4
    max_steps: int = 1000000
    mlflow: bool = False
    num_workers: int = 8
    optimizer: str = "Adam"
    resume_path: str | None = None
    scheduler: str = "linear_warmup_decay"
    precision: str = "16-mixed"
    project: str = "audiobox-sound"
    tracking_uri: str = "file:logs"
    weight_average: bool = False
