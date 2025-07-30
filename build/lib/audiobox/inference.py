from pathlib import Path

import numpy as np
import torch
from einops import repeat
from msclap import CLAP
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer
from vocos import get_voco

from model.module import AudioBoxModule
from torchode.interface import solve_ivp


class Infer:
    def __init__(self, path: Path):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = AudioBoxModule.load_from_checkpoint(path).to(self.device)
        self.model.eval()
        self.voco = get_voco(self.model.voco_type).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.tokenizer.padding_side = "right"

        self.clap = CLAP(version="2023", use_cuda=torch.cuda.is_available())

        self.steps = 64
        self.alpha = 3.0
        self.clap_audio_len = 7 * self.model.sampling_rate

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def encode_text(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        batch_encoding = self.tokenizer(
            [text + self.tokenizer.eos_token for text in texts],
            add_special_tokens=False,
            return_tensors="pt",
            max_length=127,
            truncation="longest_first",
            padding="max_length",
        )
        phoneme = batch_encoding.input_ids.to(self.device)
        phoneme_mask = batch_encoding.attention_mask.to(self.device) > 0
        phoneme_emb = self.model.t5(
            input_ids=phoneme, attention_mask=phoneme_mask
        ).last_hidden_state

        return phoneme_emb, phoneme_mask

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def clap_rank(self, audios: Tensor, texts: list[str]) -> Tensor:
        audios = audios[:, : self.clap_audio_len].mean(dim=-1)
        audios = audios.float()
        text_embed = self.clap.get_text_embeddings(texts)
        audio_embed = self.clap.clap.audio_encoder(audios)[0]

        similarity = F.cosine_similarity(text_embed, audio_embed)
        args = torch.argsort(similarity, dim=0, descending=True)
        return args

    @torch.no_grad()
    @torch.autocast(device_type="cuda")
    def generate(
        self, texts: list[str], dur: float, cutoff: int = 5
    ) -> list[np.ndarray]:
        phoneme_emb, phoneme_mask = self.encode_text(texts)
        batch_size = phoneme_emb.shape[0]

        target_len = round(self.model.sampling_rate * dur)
        latent_len = self.voco.encode_length(target_len)
        audio_mask = torch.ones(
            batch_size, latent_len, dtype=torch.bool, device=self.device
        )
        audio_context = torch.zeros(
            batch_size, latent_len, self.voco.latent_dim, device=self.device
        )

        if latent_len < 192:
            audio_mask = F.pad(audio_mask, (0, 192 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 192 - latent_len))
        elif 192 < latent_len < 384:
            audio_mask = F.pad(audio_mask, (0, 384 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 384 - latent_len))
        elif 384 < latent_len < 768:
            audio_mask = F.pad(audio_mask, (0, 768 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 768 - latent_len))
        elif 768 < latent_len < 1536:
            audio_mask = F.pad(audio_mask, (0, 1536 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 1536 - latent_len))

        def fn(t: Tensor, y: Tensor):
            out = self.model.audiobox.cfg(
                w=y,
                context=audio_context,
                times=t,
                alpha=self.alpha,
                mask=audio_mask,
                phoneme_emb=phoneme_emb,
                phoneme_mask=phoneme_mask,
            )
            return out

        y0 = torch.randn_like(audio_context)
        t = torch.linspace(0, 1, self.steps, device=self.device)

        t = repeat(t, "n -> b n", b=batch_size)
        sol = solve_ivp(
            torch.compile(fn, dynamic=False),
            y0,
            t,
            method_class=self.model.torchode_method_klass,
        )
        sampled_audio = sol.ys[-1]

        sample = self.voco.decode(sampled_audio)
        sample = sample[:, :target_len]

        sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
        args = self.clap_rank(sample, texts)
        sample = sample[args]
        sample = sample[:cutoff]
        sample = sample.detach().cpu().numpy()

        return [audio for audio in sample]

    @torch.no_grad()
    @torch.autocast(device_type="cuda")
    def variation(
        self, audios: list[np.ndarray], texts: list[str], dur: float, corrupt: float
    ) -> list[np.ndarray]:
        phoneme_emb, phoneme_mask = self.encode_text(texts)
        batch_size = phoneme_emb.shape[0]

        audios = [audio / np.iinfo(audio.dtype).max for audio in audios]
        audio_tensor = torch.from_numpy(np.stack(audios, axis=0)).to(self.device)
        audio_tensor = audio_tensor.float()
        target_len = audio_tensor.shape[1]
        latent_len = self.voco.encode_length(target_len)
        audio_enc = self.voco.encode(audio_tensor)
        audio_mask = torch.ones(
            batch_size, latent_len, dtype=torch.bool, device=self.device
        )
        audio_context = torch.zeros(
            batch_size, latent_len, self.voco.latent_dim, device=self.device
        )

        if latent_len < 192:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 192 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 192 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 192 - latent_len))
        elif 192 < latent_len < 384:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 384 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 384 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 384 - latent_len))
        elif 384 < latent_len < 768:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 768 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 768 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 768 - latent_len))
        elif 768 < latent_len < 1536:
            audio_enc = F.pad(audio_enc, (0, 0, 0, 1536 - latent_len))
            audio_mask = F.pad(audio_mask, (0, 1536 - latent_len))
            audio_context = F.pad(audio_context, (0, 0, 0, 1536 - latent_len))

        def backward(t: Tensor, y: Tensor):
            out = self.model.audiobox.cfg(
                w=y,
                context=audio_context,
                times=1 - t,
                alpha=self.alpha,
                mask=audio_mask,
                phoneme_emb=phoneme_emb,
                phoneme_mask=phoneme_mask,
            )
            return out

        t = torch.linspace(0, corrupt, self.steps, device=self.device)

        t = repeat(t, "n -> b n", b=batch_size)
        sol = solve_ivp(
            torch.compile(backward, dynamic=False),
            audio_enc,
            t,
            method_class=self.model.method #torchode_method_klass,
        )
        noised_enc = sol.ys[-1]

        corrupt_tensor = torch.tensor(1 - corrupt).to(self.device)

        def forward(t: Tensor, y: Tensor):
            out = self.model.audiobox.cfg(
                w=y,
                context=audio_context,
                times=t + corrupt_tensor,
                alpha=self.alpha,
                mask=audio_mask,
                phoneme_emb=phoneme_emb,
                phoneme_mask=phoneme_mask,
            )
            return out

        t = torch.linspace(0, corrupt, self.steps, device=self.device)

        t = repeat(t, "n -> b n", b=batch_size)
        sol = solve_ivp(
            torch.compile(forward, dynamic=False, disable=True),
            noised_enc,
            t,
            method_class=self.model.method #torchode_method_klass,
        )
        sampled_audio = sol.ys[-1]

        sample = self.voco.decode(sampled_audio)
        new_target_len = round(self.model.sampling_rate * dur)
        sample = sample[:, :new_target_len]

        sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
        sample = sample.detach().cpu().numpy()

        return [audio for audio in sample]


if __name__ == "__main__":
    infer = Infer(Path("new-stage-1.ckpt"))
    audios = infer.generate(["cat meow"] * 5, 2.0)
    audios = infer.generate(["cat meow"] * 5, 2.0)
    audios = infer.variation(audios, ["cat meow"] * 5, 2.0, 1.0)
    audios = infer.generate(["cat meow"] * 5, 4.0)
    audios = infer.generate(["cat meow"] * 5, 4.0)
    audios = infer.variation(audios, ["cat meow"] * 5, 4.0, 1.0)
    audios = infer.generate(["cat meow"] * 5, 8.0)
    audios = infer.generate(["cat meow"] * 5, 8.0)
    audios = infer.variation(audios, ["cat meow"] * 5, 8.0, 1.0)
    audios = infer.generate(["car driving"] * 5, 16.0)
    audios = infer.generate(["car driving"] * 5, 16.0)
    audios = infer.variation(audios, ["car driving"] * 5, 16.0, 1.0)
    audios = infer.generate(["car driving"] * 5, 32.0)
    audios = infer.generate(["car driving"] * 5, 32.0)
    audios = infer.variation(audios, ["car driving"] * 5, 32.0, 1.0)
    audios = infer.generate(["forest ambience"] * 5, 64.0)
    audios = infer.generate(["forest ambience"] * 5, 64.0)
    audios = infer.variation(audios, ["forest ambience"] * 5, 64.0, 1.0)
    audios = infer.generate(["cat meow"] * 5, 2.0)
    audios = infer.variation(audios, ["cat meow"] * 5, 2.0, 1.0)
    audios = infer.generate(["cat meow"] * 5, 4.0)
    audios = infer.variation(audios, ["cat meow"] * 5, 4.0, 1.0)
    audios = infer.generate(["cat meow"] * 5, 8.0)
    audios = infer.variation(audios, ["cat meow"] * 5, 8.0, 1.0)
    audios = infer.generate(["car driving"] * 5, 16.0)
    audios = infer.variation(audios, ["car driving"] * 5, 16.0, 1.0)
    audios = infer.generate(["car driving"] * 5, 32.0)
    audios = infer.variation(audios, ["car driving"] * 5, 32.0, 1.0)
    audios = infer.generate(["forest ambience"] * 5, 64.0)
    audios = infer.variation(audios, ["forest ambience"] * 5, 64.0, 1.0)
