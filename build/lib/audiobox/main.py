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

    with (
        contextlib.nullcontext()
        if cfg.train.compile or True
        else install_import_hook(
            ["data", "model", "torchode", "utils", "vocos"],
            typechecker="beartype.beartype",
        )
    ):
        from data.datamodule import AudioDataModule
        from model.module import AudioBoxModule

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    paths = OmegaConf.to_object(cfg.data.paths)
    assert isinstance(paths, list)
    assert is_bearable(paths, list[Path])
    voco = get_voco(cfg.voco)

    datamodule = AudioDataModule(
        dataset_paths=paths,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sampling_rate=voco.sampling_rate,
        channel=voco.channel,
        max_audio_len=cfg.data.max_audio_len,
        max_txt_len=cfg.data.max_txt_len,
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

    if cfg.train.base_path:
        state_dict = load_file(cfg.train.base_path)
        diff_keys = model.load_state_dict(state_dict, strict=False)
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

    model.example_input_array = (
        torch.randn(cfg.train.batch_size, cfg.data.max_audio_len, voco.latent_dim),
        torch.ones(cfg.train.batch_size, cfg.data.max_audio_len, dtype=torch.bool),
        torch.randint(
            0, model.t5.config.vocab_size, (cfg.train.batch_size, cfg.data.max_txt_len)
        ),
        torch.ones(cfg.train.batch_size, cfg.data.max_txt_len, dtype=torch.bool),
    )

    Path("logs").mkdir(exist_ok=True)
    if cfg.train.fast_dev_run:
        logger = None
        run_name = None
    elif cfg.train.mlflow:
        logger = MLFlowLogger(
            experiment_name=cfg.train.project,
            save_dir="./logs",
            tracking_uri=cfg.train.tracking_uri,
        )
        run_id = logger.run_id
        if run_id is None:
            run_name = None
        else:
            run = mlflow.get_run(run_id)
            run_name = run.info.run_name
            run = mlflow.start_run(run_id=run_id, log_system_metrics=True)
    else:
        logger = TensorBoardLogger(save_dir="logs", name=cfg.train.project)
        run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    if run_name is not None:
        checkpoint_dir = Path("checkpoints") / str(run_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = None

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "model.txt", "w") as f:
            f.write(str(model))

    if cfg.train.compile:
        model = cast(LightningModule, torch.compile(model, dynamic=False))

    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step:07d}-{val/loss:.4f}",
        monitor="val/loss",
        save_top_k=3,
        mode="min",
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [model_checkpoint]

    if cfg.train.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
                strict=False,
                check_finite=False,
            )
        )

    if cfg.train.scheduler != "None":
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if cfg.train.weight_average:

        def avg_fn(
            averaged_model_parameter: Tensor,
            model_parameter: Tensor,
            _num_averaged: Tensor,
        ) -> Tensor:
            return averaged_model_parameter * 0.99 + model_parameter * 0.01

        callbacks.append(
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
        log_every_n_steps=10,
        max_steps=cfg.train.max_steps,
        val_check_interval=0.2,
        num_sanity_val_steps=10,
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
