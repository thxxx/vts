import contextlib
import time
from pathlib import Path
from typing import cast
import hydra
import mlflow
import torch
import torch.distributed
from beartype.door import is_bearable
from hydra.core.config_store import ConfigStore
from jaxtyping import install_import_hook
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import Tensor
from vocos import get_voco

from config.config import Config, dict_to_config
from config.data.config import DataConfig
from config.model.config import (
    AudioBoxGigaConfig,
    AudioBoxLargeConfig,
    AudioBoxMediumConfig,
    AudioBoxSmallConfig,
)
from config.train.config import TrainConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="train", name="base", node=TrainConfig)
cs.store(group="data", name="base", node=DataConfig)
cs.store(group="audiobox", name="giga", node=AudioBoxGigaConfig)
cs.store(group="audiobox", name="large", node=AudioBoxLargeConfig)
cs.store(group="audiobox", name="medium", node=AudioBoxMediumConfig)
cs.store(group="audiobox", name="small", node=AudioBoxSmallConfig)


@hydra.main(config_name="config", version_base=None)
def main(_cfg: DictConfig):
    torch._dynamo.config.optimize_ddp = False
    seed_everything(42, workers=True)
    cfg = dict_to_config(_cfg)
    print(f"\n\nConfig : {cfg}\n\n")

    with (
        contextlib.nullcontext() # 아무 것도 하지 않고, with의 조건문으로 사용가능한 문법
        if cfg.train.compile or True # 근데 어짜피 True.
        else install_import_hook(
            ["data", "model", "torchode", "utils", "vocos"],
            typechecker="beartype.beartype",
        )
    ):
        from data.datamodule import AudioDataModule
        from model.module_voice import AudioBoxModule

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high") # high = 정밀도와 속도의 균형. default 값

    paths = OmegaConf.to_object(cfg.data.paths) # OmegaConf 객체를 Python 객체로 변환해줌.
    assert isinstance(paths, list)
    assert is_bearable(paths, list[Path])

    voco = get_voco(cfg.voco)
    dataset_path = Path("./total_voice_0409.csv")
    
    datamodule = AudioDataModule(
        dataset_path=dataset_path,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sampling_rate=voco.sampling_rate,
        channel=voco.channel,
        max_audio_len=cfg.data.max_audio_len,
        max_txt_len=cfg.data.max_txt_len,
        max_video_len=(24//8)*10+1
    )

    model = AudioBoxModule(
        dim=cfg.audiobox.dim,
        depth=cfg.audiobox.depth,
        heads=cfg.audiobox.heads,
        attn_dropout=cfg.audiobox.attn_dropout,
        ff_dropout=cfg.audiobox.ff_dropout,
        kernel_size=cfg.audiobox.kernel_size,
        voco_type=cfg.voco,
        optimizer=cfg.train.optimizer,
        lr=cfg.train.lr,
        scheduler=cfg.train.scheduler,
        max_audio_len=cfg.data.max_audio_len,
        max_steps=cfg.train.max_steps,
        text_repo_id=cfg.audiobox.text_repo_id,
    )

    if cfg.train.base_path: # 이어서 학습하는 경우
        state_dict = load_file(cfg.train.base_path)
        diff_keys = model.audiobox.load_state_dict(state_dict, strict=False)
        if diff_keys.unexpected_keys:
            keys = "\n".join(
                key for key in diff_keys.unexpected_keys if not key.startswith("voco")
            )
            if keys:
                raise ValueError(
                    f"{len(diff_keys.unexpected_keys)} unexpected keys detected!\n"
                    f"Full list: {keys}"
                )
    else:
        print("Warning: missing base path. Are you sure?")

    # PyTorch Lightning에서 사용하는 example_input_array. model의 forward 입력값을 확실히 알려주는 용도
    # 디버깅, ONNX export, 모델 구조 추론 등에 사용됨. 정확히 왜 한다는건지 아직 잘 모르겠다.
    model.example_input_array = (
        torch.randn(cfg.train.batch_size, cfg.data.max_audio_len, voco.latent_dim), # latent
        torch.ones(cfg.train.batch_size, cfg.data.max_audio_len, dtype=torch.bool), # latent mask
        torch.randint(
            0, model.t5.config.vocab_size, (cfg.train.batch_size, cfg.data.max_txt_len) # text input ids
        ),
        torch.ones(cfg.train.batch_size, cfg.data.max_txt_len, dtype=torch.bool), # text attention mask
        torch.randn(cfg.train.batch_size, cfg.data.max_audio_len, 12), # voice condition
    )

    # 로깅방식
    Path("logs").mkdir(exist_ok=True) # logs 폴더를 만드는데 이미 있어도 에러없이 지나간다.
    if cfg.train.fast_dev_run:
        logger = None
        run_name = None
    elif cfg.train.mlflow:
        mlflow.end_run()
        print("\n\n====MLFLOW 실행====\n\n")
        logger = MLFlowLogger(
            experiment_name=cfg.train.project,
            save_dir="./mllogs",
            tracking_uri=cfg.train.tracking_uri,
        )
        run_id = logger.run_id
        print(f"\n\nrun_id : {run_id}")
        if run_id is None:
            run_name = None
        else:
            run = mlflow.get_run(run_id)
            run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
            # run_name = run.info.run_name
            run = mlflow.start_run(run_id=run_id, log_system_metrics=True)
    else:
        logger = TensorBoardLogger(save_dir="logs", name=cfg.train.project)
        run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    if run_name is not None:
        checkpoint_dir = Path("checkpoints") / str(run_name) # 나누기가 아니라 dir 추가하는거구나
        checkpoint_dir.mkdir(parents=True, exist_ok=True) # 폴더가 여러개 필요해도 만든다.
    else:
        checkpoint_dir = None

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "model.txt", "w") as f:
            f.write(str(model))

    if cfg.train.compile: # False가 기본값이긴 함
        model = cast(LightningModule, torch.compile(model, dynamic=False)) # dynamic=False : 입력이 고정되어 있다고 가정, 최적화가 더 잘된다. 자주 바뀌면 오히려 비효율적. cast는 타입을 LightningModule로 유지하기 위함

    # Pytorch lightning의 callback들 : 자동으로 특정 타이밍에 실행되는 로직.
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step:07d}-{val/loss:.4f}",
        monitor="val/loss", # 이 metric을 기준으로
        # every_n_train_steps=5000,
        # save_top_k=5,       # 가장 좋은 모델 3개 저장
        # mode="min",         # 낮을 수록 좋다.
        auto_insert_metric_name=False,
        save_top_k=-1,           # 모든 체크포인트 저장
        every_n_train_steps=5000,  # 1000 스텝마다 저장
        save_on_train_epoch_end=False,  # 스텝 단위로 저장할 거면 이걸 False로
    )
    callbacks: list[Callback] = [model_checkpoint]

    # 이 콜백들은 Trainer에 callbacks=[...] 식으로 넣어주면 자동으로 동작
    if cfg.train.early_stop: # False
        callbacks.append(
            EarlyStopping(
                monitor="val/loss", # 이 metric을 기준으로
                min_delta=0.00,
                patience=3,         # 3 epoch 동안 개선이 없으면 스탑
                verbose=False,
                mode="min",
                strict=False,
                check_finite=False,
            )
        )

    if cfg.train.scheduler != "None":
        callbacks.append(LearningRateMonitor(logging_interval="step")) # lr도 step마다 로깅해라!

    if cfg.train.weight_average: # False
        def avg_fn(
            averaged_model_parameter: Tensor,
            model_parameter: Tensor,
            _num_averaged: Tensor,
        ) -> Tensor:
            return averaged_model_parameter * 0.99 + model_parameter * 0.01

        callbacks.append(
            # overfitting 방지 및 일반화 성능을 높여주는 테크닉
            StochasticWeightAveraging(
                swa_lrs=cfg.train.lr / 10,
                swa_epoch_start=0.5,
                annealing_epochs=10,
                annealing_strategy="cos",
                avg_fn=avg_fn,
            )
        )

    precision = cfg.train.precision
    assert precision == "16-mixed" or precision == "32" or precision == "bf16-mixed"

    trainer = Trainer(
        strategy="ddp",
        accumulate_grad_batches=cfg.train.acc,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks,
        detect_anomaly=cfg.train.fast_dev_run,
        fast_dev_run=cfg.train.fast_dev_run,
        logger=logger,
        log_every_n_steps=1000, # original : 10
        max_steps=cfg.train.max_steps, # default : 1M
        val_check_interval=0.2,
        num_sanity_val_steps=5,
        precision=precision,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.train.resume_path)

    if checkpoint_dir is None:
        save_path = "model.ckpt"
    else:
        save_path = checkpoint_dir / "model.ckpt"

    if cfg.train.fast_dev_run:
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.test(ckpt_path="best", datamodule=datamodule)
        trainer.save_checkpoint(save_path, weights_only=True)

    if cfg.train.mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
