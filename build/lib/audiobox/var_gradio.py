import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from einops import repeat
from msclap import CLAP
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer
from vocos import get_voco

from model.module import AudioBoxModule
from torchode.interface import solve_ivp


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
def clap_rank(audios: Tensor, texts: list[str]) -> Tensor:
    clap_audio_len = 7 * model.sampling_rate
    audios = audios[:, :clap_audio_len].mean(dim=-1)
    audios = audios.float()
    text_embed = clap_model.get_text_embeddings(texts)
    audio_embed = clap_model.clap.audio_encoder(audios)[0]

    similarity = F.cosine_similarity(text_embed, audio_embed)
    args = torch.argsort(similarity, dim=0, descending=True)
    print(similarity[args])
    return args


@torch.no_grad()
def create_audio(
    audio_tuple: tuple[int, np.ndarray],
    text: str,
    alpha: float,
    length: float,
    corrupt: float,
    method: str,
    steps: int,
    prog=gr.Progress(),
):
    num_sample = 1
    texts = [text]

    batch_encoding = tokenizer(
        [text + tokenizer.eos_token for text in texts],
        add_special_tokens=False,
        return_tensors="pt",
        max_length=127,
        truncation="longest_first",
        padding="max_length",
    )
    phoneme = batch_encoding.input_ids.to(device)
    phoneme_mask = batch_encoding.attention_mask.to(device) > 0

    sr, audio = audio_tuple
    audio = audio / np.iinfo(audio.dtype).max
    audio_tensor = torch.from_numpy(audio).to(device)
    audio_tensor = audio_tensor.float()
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.T, sr, model.sampling_rate
    ).T
    audio_tensor = audio_tensor.unsqueeze(dim=0)
    audio_enc = voco.encode(audio_tensor)

    latent_len = audio_enc.shape[1]
    audio_mask = torch.ones(1, latent_len, dtype=torch.bool, device=device)
    audio_context = torch.zeros(
        num_sample, audio_mask.shape[1], voco.latent_dim, device=device
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

    audio_mask = repeat(audio_mask, "() ... -> b ...", b=num_sample)
    phoneme = repeat(phoneme, "() ... -> b ...", b=num_sample)
    phoneme_mask = repeat(phoneme_mask, "() ... -> b ...", b=num_sample)

    with torch.autocast(device_type=model.device.type, enabled=False):
        phoneme_emb = model.t5(
            input_ids=phoneme, attention_mask=phoneme_mask
        ).last_hidden_state

    def backward(t: Tensor, y: Tensor):
        out = -model.audiobox.cfg(
            w=y,
            context=audio_context,
            times=1 - t,
            alpha=alpha,
            mask=audio_mask,
            phoneme_emb=phoneme_emb,
            phoneme_mask=phoneme_mask,
        )
        return out

    t = torch.linspace(0, corrupt, steps, device=device)
    t = repeat(t, "n -> b n", b=num_sample)

    sol = solve_ivp(
        torch.compile(backward, dynamic=False),
        audio_enc,
        t,
        method_class=method,
        prog=prog.tqdm([None] * (steps - 1)),
    )
    noised_enc = sol.ys[-1]

    def forward(t: Tensor, y: Tensor):
        out = model.audiobox.cfg(
            w=y,
            context=audio_context,
            times=t + 1 - corrupt,
            alpha=alpha,
            mask=audio_mask,
            phoneme_emb=phoneme_emb,
            phoneme_mask=phoneme_mask,
        )
        return out

    t = torch.linspace(0, corrupt, steps, device=device)

    t = repeat(t, "n -> b n", b=num_sample)
    sol = solve_ivp(
        torch.compile(forward, dynamic=False),
        noised_enc,
        t,
        method_class=method,
        prog=prog.tqdm([None] * (steps - 1)),
    )
    sampled_audio = sol.ys[-1]

    target_len = round(model.sampling_rate * length)
    sample = voco.decode(sampled_audio)
    sample = sample[:, :target_len]
    sample = sample / sample.abs().amax(dim=1, keepdim=True).clamp_min(1)
    best_sample = sample[0].detach().cpu().numpy()
    best_sample_int16 = (best_sample * np.iinfo(np.int16).max).astype(np.int16)

    return (model.sampling_rate, best_sample_int16)


# Launch the app
if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    model = AudioBoxModule.load_from_checkpoint(args.path).to(device)
    voco = get_voco(model.voco_type).to(device)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tokenizer.padding_side = "right"

    clap_model = CLAP(version="2023", use_cuda=True)

    # Default values for parameters
    default_text = "Bright ringtone."
    default_alpha = 3.0
    default_length = 5.0  # seconds

    # Gradio app interface
    interface = gr.Interface(
        fn=create_audio,
        inputs=[
            gr.Audio(label="Input Audio"),
            gr.Textbox(label="Text Input", value=default_text),
            gr.Slider(
                minimum=0.0, maximum=5.0, step=0.5, value=default_alpha, label="CFG"
            ),
            gr.Slider(
                minimum=0.5,
                maximum=60.0,
                step=0.5,
                value=default_length,
                label="Length (seconds)",
            ),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Corrupt"),
            gr.Radio(["tsit5", "midpoint"], value="tsit5", label="ODE Solver"),
            gr.Slider(
                minimum=16, maximum=64, step=16, value=64, label="Tsit5 Solver steps"
            ),
        ],
        outputs=gr.Audio(label="Generated Stereo Audio"),
        title="Audio Variation",
    )

    interface.launch(server_name="0.0.0.0", share=args.share)
