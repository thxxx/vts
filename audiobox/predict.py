import argparse
from pathlib import Path

import torch
import torchaudio
from einops import repeat
from transformers import AutoTokenizer

from model.module import AudioBoxModule


@torch.no_grad
def main(model_path: Path, text: str, length: float):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    model = AudioBoxModule.load_from_checkpoint(model_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokenizer.padding_side = "right"

    phoneme = tokenizer.encode(
        text + tokenizer.eos_token, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    audio_len = model.voco.decode_length(model.max_audio_len)
    if length < 0:
        target_len = audio_len
    else:
        target_len = min(round(model.sampling_rate * length), audio_len)
    audio_context = torch.zeros(1, audio_len, model.voco.channel).to(device)
    audio_mask = torch.ones(1, target_len, dtype=torch.bool).to(device)
    audio_mask = torch.nn.functional.pad(audio_mask, pad=(0, audio_len - target_len))
    phoneme_mask = torch.ones_like(phoneme, dtype=torch.bool)
    audio_context = repeat(audio_context, "() ... -> b ...", b=1)
    audio_mask = repeat(audio_mask, "() ... -> b ...", b=1)
    phoneme = repeat(phoneme, "() ... -> b ...", b=1)
    phoneme_mask = repeat(phoneme_mask, "() ... -> b ...", b=1)
    phoneme = torch.nn.functional.pad(phoneme, pad=(0, 128 - phoneme.shape[1]))
    phoneme_mask = torch.nn.functional.pad(
        phoneme_mask, pad=(0, 128 - phoneme_mask.shape[1])
    )

    sample = model.sample(
        audio=audio_context,
        audio_mask=audio_mask,
        phoneme=phoneme,
        phoneme_mask=phoneme_mask,
        alpha=3.0,
    )
    sample = sample[0, :target_len].detach().cpu()

    torchaudio.save(
        "sample.wav", sample, sample_rate=model.sampling_rate, channels_first=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--length", type=float, required=False, default=-1)

    args = parser.parse_args()

    main(model_path=args.path, text=args.text, length=args.length)
