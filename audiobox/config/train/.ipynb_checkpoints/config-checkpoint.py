from dataclasses import dataclass


@dataclass
class TrainConfig:
    acc: int = 2
    batch_size: int = 32
    base_path: str = "/workspace/new-stage-2.ckpt"
    compile: bool = False
    early_stop: bool = False
    fast_dev_run: bool = False
    gradient_clip_val: float = 0.2
    lr: float = 5e-6
    # lr: float = 2e-5
    max_steps: int = 40000
    # max_steps: int = 100000
    mlflow: bool = True
    num_workers: int = 4
    optimizer: str = "AdamW"
    resume_path: str | None = None
    scheduler: str = "linear_warmup_decay"
    # precision: str = "16-mixed"
    precision: str = "32"
    project: str = "audiobox-video-clips"
    tracking_uri: str = "file:mllogs"
    weight_average: bool = False
    max_video_latent_len:int = 64
    video_latent_dim:int = 1280
    video_factor:int = 32
    video_resolution:int = 128
