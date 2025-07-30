from dataclasses import dataclass


@dataclass
class TrainConfig:
    acc: int = 1
    batch_size: int = 4
    base_path: str = "./audiobox_voice_0415.safetensors"
    compile: bool = False
    early_stop: bool = False
    fast_dev_run: bool = False
    gradient_clip_val: float = 0.2
    lr: float = 2e-4
    max_steps: int = 1000000
    mlflow: bool = True
    num_workers: int = 8
    optimizer: str = "Adam"
    # resume_path: str | None = None
    resume_path: str | None = '/home/khj6051/alignment-v3/audiobox/0055000-0.6663.ckpt'
    scheduler: str = "linear_warmup_decay"
    # precision: str = "16-mixed"
    precision: str = "32"
    project: str = "audiobox-sound"
    tracking_uri: str = "file:mllogs"
    weight_average: bool = False
