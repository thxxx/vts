import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import random

import numpy as np
import torch
import torchaudio
from einops import rearrange, repeat
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchdiffeq import odeint
from transformers import AutoTokenizer, T5EncoderModel
from vocos import get_voco

from model.audiobox_video import AudioBox
from model.loss import masked_loss
from torchode.interface import solve_ivp
from utils.mask import min_span_mask, prob_mask_like
from utils.typing import AudioTensor, EncMaskTensor, EncTensor, LossTensor
from utils.utils import plot_with_cmap, write_html, get_dynamic

class AudioBoxModule(LightningModule):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        kernel_size: int,
        voco_type: str,
        max_audio_len: int,
        optimizer: str = "Adam",
        lr: float = 1e-4,
        scheduler: str = "linear_warmup_decay",
        use_torchode=True,
        torchdiffeq_ode_method="midpoint",
        torchode_method_klass="tsit5",
        max_steps: int = 1000000,
        text_repo_id: str = "google/flan-t5-base",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.voco_type = voco_type
        print("voco type  : ", voco_type)
        voco = get_voco(self.voco_type)

        self.sampling_rate = voco.sampling_rate

        self.t5 = T5EncoderModel.from_pretrained(text_repo_id)
        self.t5.eval()
        for param in self.t5.parameters():
            param.requires_grad_(False)
        text_dim = self.t5.config.d_model

        self.audiobox = AudioBox(
            audio_dim=voco.latent_dim,
            text_dim=text_dim,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            kernel_size=kernel_size,
        )
        self.mask_fracs = (0.7, 1.0)
        self.min_span = 10
        self.drop_prob = 0.4
        self.max_audio_len = max_audio_len

        self.use_torchode = use_torchode
        self.torchode_method_klass = torchode_method_klass

        self.steps = 64
        self.sigma = 1e-5
        self.method = torchdiffeq_ode_method

        self.optim = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.max_steps = max_steps

        self.debug_logger = logging.getLogger("audiobox")
        self.debug_logger.setLevel(logging.DEBUG)
        try:
            output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            output_dir = "outputs"
        if not Path(output_dir).exists():
            Path(output_dir).mkdir()
        handler = logging.FileHandler(f"{output_dir}/audiobox.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.debug_logger.addHandler(handler)

    def solve(
        self,
        context: EncTensor,
        mask: EncMaskTensor,
        text: Tensor,
        text_mask: Tensor,
        video_latent: Tensor,
        alpha=0.0,
    ) -> EncTensor:
        with torch.autocast(device_type=self.device.type, enabled=False):
            text_emb = self.t5(
                input_ids=text, attention_mask=text_mask
            ).last_hidden_state

        def fn(t: Float[Tensor, "..."], y: Float[Tensor, "..."]):
            out = self.audiobox.cfg(
                w=y,
                context=context,
                times=t,
                alpha=alpha,
                mask=mask,
                text_emb=text_emb,
                text_mask=text_mask,
                video_latent=video_latent
            )
            return out

        y0 = torch.randn_like(context)
        t = torch.linspace(0, 1, self.steps, device=self.device)

        if self.use_torchode:
            batch = context.shape[0]
            t = repeat(t, "n -> b n", b=batch)
            sol = solve_ivp(
                torch.compile(fn, dynamic=False),
                y0,
                t,
                method_class=self.torchode_method_klass,
            )
            sampled = sol.ys[-1]
        else:
            trajectory = odeint(
                fn,
                y0,
                t,
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=1 / self.steps),
            )
            sampled = trajectory[-1]

        return sampled

    @torch.compiler.disable
    @torch.no_grad
    def get_span_mask(self, audio_mask: EncMaskTensor):
        audio_lens = audio_mask.sum(dim=1).detach().cpu().numpy()
        mask_len = audio_mask.shape[-1]
        span_mask = pad_sequence(
            [
                torch.from_numpy(
                    min_span_mask(
                        int(audio_len),
                        fmin=self.mask_fracs[0],
                        fmax=self.mask_fracs[1],
                        min_span=self.min_span,
                    )
                ).to(self.device)
                for audio_len in audio_lens
            ],
            batch_first=True,
        )
        return F.pad(span_mask, (0, self.max_audio_len - span_mask.shape[1]))
    
    def sample(
        self,
        audio_enc: AudioTensor,
        audio_mask: EncMaskTensor,
        text: Tensor,
        text_mask: Tensor,
        video_latent: AudioTensor,
        alpha=0.0,
    ):
        span_mask = self.get_span_mask(audio_mask)
        span_mask = torch.ones_like(span_mask)

        audio_context = torch.where(rearrange(span_mask, "b l -> b l ()"), 0, audio_enc)

        sampled_audio_enc = self.solve(
            audio_context, audio_mask, text, text_mask, video_latent=video_latent, alpha=alpha
        )
        return sampled_audio_enc

    def forward(
        self,
        audio_enc: AudioTensor,
        audio_mask: EncMaskTensor,
        text: Tensor,
        text_mask: Tensor,
        video_latent: AudioTensor
    ) -> LossTensor:
        try:
            batch = audio_enc.shape[0]

            with torch.no_grad():
                span_mask = self.get_span_mask(audio_mask)
                with torch.autocast(device_type=self.device.type, enabled=False):
                    text_emb = self.t5(
                        input_ids=text, attention_mask=text_mask
                    ).last_hidden_state

            audio_x0 = torch.randn_like(audio_enc)

            times = torch.rand((batch,), dtype=audio_enc.dtype, device=self.device) # torch.rand는 0~1 uniform sampling
            t = rearrange(times, "b -> b () ()")
            w = (1 - (1 - self.sigma) * t) * audio_x0 + t * audio_enc

            cond_drop_mask = prob_mask_like((batch, 1), self.drop_prob, self.device)
            audio_cond_mask = span_mask | cond_drop_mask

            audio_context = torch.where(
                rearrange(audio_cond_mask, "b l -> b l ()"), 0, audio_enc
            )

            text_drop_mask = prob_mask_like((batch,), self.drop_prob, self.device)
            text_emb = torch.where(
                rearrange(text_drop_mask, "b -> b () ()"), 0, text_emb
            )

            pred_audio_flow = self.audiobox(
                w=w,
                times=times,
                audio_mask=audio_mask,
                context=audio_context,
                text_emb=text_emb,
                text_mask=text_mask,
                video_latent=video_latent
            )

            target_audio_flow = audio_enc - (1 - self.sigma) * audio_x0
            loss = masked_loss(pred_audio_flow, target_audio_flow, audio_cond_mask, "mse")

            # NaN 체크
            if loss is None or torch.isnan(loss):
                self.print(f"[step {self.global_step}] ❌ NaN loss detected, skipping step.")
                dummy = pred_audio_flow.sum() * 0
                return dummy

            return loss

        except Exception as e:
            self.print(f"[step {self.global_step}] ❌ Exception in training_step: {e}")
            return None  # 이 step은 스킵됨

    @torch.compiler.disable
    def log_loss(self, id: str, loss: Tensor | float, train: bool):
        if train:
            self.log(id, loss, on_step=True, on_epoch=False, logger=True)
        else:
            self.log(id, loss, on_step=False, on_epoch=True, sync_dist=True, logger=True)

    def single_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], prefix: str
    ) -> LossTensor:
        audio, video_latent, audio_mask, text, text_mask = batch
        loss = self(audio, audio_mask, text, text_mask, video_latent=video_latent)
        train = prefix == "train"

        self.log_loss(f"{prefix}/loss", loss, train)
        return loss

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        return self.single_step(batch, "train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        self.single_step(batch, "val")
        if batch_idx < 5:
            self.log_table(batch, "val", batch_idx)
            # self.validate_generation(batch, batch_idx)
    
    def validate_generation(self, batch, idx):
        audios = [
            './voice_samples/piung.wav',
            './voice_samples/beepbeep.m4a',
            './voice_samples/charging.m4a',
        ]
        voices = [
            './voice_samples/piung_voice.npy',
            './voice_samples/beepbeep_voice.npy',
            './voice_samples/charging_voice.npy',
        ]
        texts = [
            './voice_samples/piung_token.npy',
            './voice_samples/beepbeep_token.npy',
            './voice_samples/charging_token.npy',
        ]
        text_masks = [
            './voice_samples/piung_token_mask.npy',
            './voice_samples/beepbeep_token_mask.npy',
            './voice_samples/charging_token_mask.npy',
        ]
        audio_len = [
            int(3.7*21.5),
            int(3*21.5),
            int(4*21.5),
        ]

        waveform, sr = torchaudio.load(audios[idx])
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sampling_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        dynamic_context = get_dynamic(waveform, 400).unsqueeze(dim=0).to(self.device)

        audio_enc = batch[0][0].unsqueeze(dim=0).to(self.device)
        audio_mask = torch.cat((
            torch.ones((1, audio_len[idx])),
            torch.zeros((1, self.max_audio_len - audio_len[idx]))
        ), dim=-1) > 0
        audio_mask = audio_mask.to(self.device)
        # audio_mask = batch[1][0].unsqueeze(dim=0).to(self.device) # 1, 400
        text_embed = torch.from_numpy(np.load(texts[idx])).unsqueeze(dim=0).to(self.device)
        text_mask = torch.from_numpy(np.load(text_masks[idx])).unsqueeze(dim=0).to(self.device)

        self.log_table(
            (audio_enc, audio_mask, text_embed, text_mask, dynamic_context),
            'val',
            idx
        )

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        self.single_step(batch, "test")
        if batch_idx < 5:
            self.log_table(batch, "test", batch_idx)

    @rank_zero_only
    @torch.no_grad
    def log_table(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], prefix: str, batch_idx: int
    ):
        audio_enc, video_latent, audio_mask, text, text_mask = batch
        random_index = torch.randint(0, audio_enc.shape[0], (1,))
        random_audio_mask = audio_mask[[random_index]]
        random_audio_enc = audio_enc[[random_index]]
        random_text = text[[random_index]]
        random_text_mask = text_mask[[random_index]]
        rondom_video_latent = video_latent[[random_index]]

        gen_audio_enc = self.sample(
            audio_enc=torch.zeros_like(random_audio_enc),
            audio_mask=random_audio_mask,
            text=random_text,
            text_mask=random_text_mask,
            video_latent=rondom_video_latent,
            alpha=1.0,
        )

        span_mask = self.get_span_mask(random_audio_mask)

        context = torch.where(
            rearrange(span_mask, "b l -> b l ()"), 0, random_audio_enc
        )
        pred_audio_enc = self.solve(
            context=context,
            mask=random_audio_mask,
            text=random_text,
            text_mask=random_text_mask,
            video_latent=rondom_video_latent
        )
        pred_audio_enc = torch.where(
            rearrange(span_mask, "b l -> b l ()"),
            pred_audio_enc,
            random_audio_enc,
        )
        self.log_data(
            random_audio_enc,
            pred_audio_enc,
            context,
            gen_audio_enc,
            random_text,
            random_audio_mask,
            prefix,
            batch_idx,
            rondom_video_latent
        )

    @torch.compiler.disable
    def log_data(
        self,
        random_audio_enc: Tensor,
        pred_audio_enc: Tensor,
        cond_audio_enc: Tensor,
        gen_audio_enc: Tensor,
        random_text: Tensor,
        random_audio_mask: Tensor,
        prefix: str,
        batch_idx: int,
        rondom_voice_enc: Tensor
    ):
        data: list[tuple[np.ndarray, np.ndarray]] = []
        mel_voco = get_voco("mel").to(self.device)
        random_audio_len = get_voco(self.voco_type).decode_length(
            int(random_audio_mask.sum().item())
        )
        voco = get_voco(self.voco_type).to(self.device)
        for enc in [random_audio_enc, pred_audio_enc, cond_audio_enc, gen_audio_enc]:
            try:
                audio = voco.decode(enc)
                audio = audio[:, :random_audio_len]
                audio = audio.float()
                mel_audio = torchaudio.functional.resample(
                    rearrange(audio, "() l c -> () c l"),
                    orig_freq=self.sampling_rate,
                    new_freq=mel_voco.sampling_rate,
                )
                mel_audio = rearrange(mel_audio, "() c l -> c l ()")
                mel = mel_voco.encode(mel_audio).detach().cpu().numpy()
                audio = audio.detach().cpu().numpy()
                audio /= np.maximum(audio.max(axis=(1, 2)), 1)
                audio_numpy = (audio[0] * np.iinfo(np.int16).max).astype(np.int16)
                data.append((audio_numpy, rearrange(mel, "c h w -> c w h")))
            except Exception as e:
                self.debug_logger.debug(
                    f"Error occured while plotting\n{e}\n", exc_info=True
                )

        if isinstance(self.logger, TensorBoardLogger):
            for (audio, _), name in zip(data, ["real", "pred", "cond", "gen"]):
                self.logger.experiment.add_audio(
                    f"{prefix}/audio/{name}",
                    (audio / np.iinfo(np.int16).max).mean(axis=-1),
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
            mel_plot = plot_with_cmap(
                list(rearrange(mel, "c w h -> (c w) h") for _, mel in data)
            )
            self.logger.experiment.add_image(
                f"{prefix}/mel",
                mel_plot,
                self.global_step,
                dataformats="HWC",
            )
        elif isinstance(self.logger, MLFlowLogger):
            assert self.logger.run_id is not None
            step = self.global_step
            audio_paths = []
            image_paths = []
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            tokenizer.padding_side = "right"
            caption = tokenizer.decode(random_text[0], skip_special_tokens=True)
            with TemporaryDirectory() as temp_dir:
                Path(temp_dir, prefix).mkdir()
                for (audio, mel), name in zip(data, ["real", "pred", "cond", "gen"]):
                    audio_path = Path(
                        temp_dir, prefix, f"{step:07d}_{batch_idx:03d}_{name}.flac"
                    )
                    torchaudio.save(
                        audio_path,
                        torch.from_numpy(audio / np.iinfo(np.int16).max),
                        self.sampling_rate,
                        channels_first=False,
                    )
                    self.logger.experiment.log_artifact(
                        self.logger.run_id, audio_path, f"{prefix}/audio"
                    )
                    audio_paths.append(audio_path)
                    mel_plot = plot_with_cmap([mel_item for mel_item in mel])
                    image_path = Path(
                        temp_dir, prefix, f"{step:07d}_{batch_idx:03d}_{name}.png"
                    )
                    plt.imsave(image_path, mel_plot)
                    self.logger.experiment.log_artifact(
                        self.logger.run_id, image_path, f"{prefix}/image"
                    )
                    image_paths.append(image_path)

                html = write_html(audio_paths, image_paths, caption)
                self.logger.experiment.log_text(
                    self.logger.run_id,
                    html,
                    f"{prefix}/{step:07d}_{batch_idx:03d}.html",
                )

    def configure_optimizers(self):
        match self.optim:
            case "Adam":
                optimizer = Adam(self.parameters(), lr=self.lr)
            case "AdamW":
                optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
            case _:
                raise ValueError(f"Unknown optimizer: {self.optim}")

        match self.scheduler:
            case "linear_warmup_decay":
                warmup_scheduler = LinearLR(
                    optimizer, start_factor=1 / 5000, end_factor=1.0, total_iters=5000
                )
                decay_scheduler = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=self.max_steps,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, decay_scheduler],
                    milestones=[5000],
                )
            case _:
                raise ValueError(f"Unknown scheduler: {self.scheduler}")

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfigType(
                scheduler=scheduler,
                interval="step",
                frequency=1,
                reduce_on_plateau=False,
                strict=True,
            ),
        )

    def on_save_checkpoint(self, checkpoint: dict[str, dict[str, Any]]):
        for key in list(checkpoint["state_dict"]):
            if key.startswith("t5"):
                del checkpoint["state_dict"][key]

    def on_load_checkpoint(self, checkpoint: dict[str, dict[str, Any]]):
        for name, param in self.t5.named_parameters():
            checkpoint["state_dict"][f"t5.{name}"] = param.data.clone()
        checkpoint["state_dict"]["t5.encoder.embed_tokens.weight"] = (
            self.t5.encoder.embed_tokens.weight.data.clone()
        )
