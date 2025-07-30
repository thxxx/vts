import base64
import json
import subprocess
from pathlib import Path

import torchaudio
import librosa
import random

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
from librosa import filters
from torch.nn import functional as F

from utils.mask import mask_from_lengths
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from utils.typing import EncTensor, LengthTensor
from scipy.signal import medfilt

plt.switch_backend("agg")


def pad_sequence(
    sequences: list[Tensor], batch_first: bool = False, padding_value: int = 0
) -> Tensor:
    """
    Pad a list of variable length Tensors with zero padding to the right.
    Return a Tensor of shape (batch, max_time, channel) if batch_first is True,
    else (max_time, batch, channel).
    The original pad_sequence function from PyTorch errors when compiling.

    Args:
        sequences: List of variable length Tensors.
        batch_first: If True, return Tensor of shape (batch, max_time, channel).
        padding_value: Value to pad with.

    Returns:
        Padded Tensor.
    """
    max_len = max([seq.size(0) for seq in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + sequences[0].size()[1:]
    else:
        out_dims = (max_len, len(sequences)) + sequences[0].size()[1:]
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor


def plot_with_cmap(mels: list[np.ndarray], sharex: bool = True):
    fig, axes = plt.subplots(len(mels), 1, figsize=(20, 8), sharex=sharex)

    if len(mels) == 1:
        axes = np.array([axes])

    im = None
    for i, mel in enumerate(mels):
        im = axes[i].imshow(mel, aspect="auto", origin="lower", interpolation="none")

    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.canvas.draw()
    plt.close(fig)

    return np.array(
        fig.canvas.buffer_rgba()  # pyright: ignore [reportAttributeAccessIssue]
    )


def normalize_audio(
    audio_enc: EncTensor, audio_lens: LengthTensor
) -> tuple[EncTensor, Tensor, Tensor]:
    """
    Normalize audio encodings to have zero mean and unit variance.
    Each audio encoding is 2-dimensional and has zero padding to the right.
    Return normalized audio encodings, mean, and standard deviation.

    Args:
        audio_enc: Audio encodings. Shape: (batch, time, channel).
        audio_lens: Lengths of audio encodings. Shape: (batch,).

    Returns:
        audio_enc: Normalized audio encodings. Shape: (batch, time, channel).
        audio_mean: Mean of audio encodings. Shape: (batch,).
        audio_std: Standard deviation of audio encodings. Shape: (batch,).
    """
    audio_mask = mask_from_lengths(audio_lens, audio_enc.shape[1])
    # audio_mean = (audio_enc.mean(dim=2) * audio_mask).sum(dim=1) / audio_lens
    # audio_sq_mean = ((audio_enc**2).mean(dim=2) * audio_mask).sum(dim=1) / audio_lens
    # nelem = audio_lens * audio_enc.shape[2]
    # bessel_correction = nelem / (nelem - 1)
    # audio_std = torch.sqrt((audio_sq_mean - audio_mean**2) * bessel_correction)
    batch_size = audio_enc.shape[0]
    audio_mean = torch.full((batch_size,), -1.430645).to(audio_enc.device)
    audio_std = torch.full((batch_size,), 2.1208718).to(audio_enc.device)
    audio_mean = rearrange(audio_mean, "b -> b () ()")
    audio_std = rearrange(audio_std, "b -> b () ()")
    normalized_audio_enc = (
        (audio_enc - audio_mean)
        / (audio_std + 1e-5)
        * rearrange(audio_mask, "b l -> b l ()")
    )
    return normalized_audio_enc, audio_mean, audio_std


def write_html(audio_paths: list[Path], image_paths: list[Path], description: str):
    html = f"""
    <html>
    <head>
        <title>Audio and Mel Preview</title>
        <!-- Lightbox2 CSS -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css" rel="stylesheet" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                margin: 0;
                padding: 0;
            }}
            .container {{
                /* Removed max-width to use full screen width */
                margin: 0 auto;
                padding: 20px;
            }}
            .description {{
                background-color: #fff;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                max-width: 1000px;
                margin-left: auto;
                margin-right: auto;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr); /* Set to 2 columns */
                grid-gap: 20px;
            }}
            .card {{
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                text-align: center;
            }}
            .card h3 {{
                margin-top: 0;
                text-transform: capitalize;
            }}
            audio {{
                width: 100%;
                margin: 10px 0;
            }}
            img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
                cursor: pointer;
                transition: transform 0.2s;
            }}
            img:hover {{
                transform: scale(1.02);
            }}
            @media (max-width: 800px) {{
                .grid {{
                    grid-template-columns: 1fr; /* Stack cards on small screens */
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="description">
                <h2>Description</h2>
                <p>{description}</p>
            </div>
            <div class="grid">
    """

    names = ["real", "pred", "cond", "gen"]
    for row_name, audio_path, image_path in zip(names, audio_paths, image_paths):
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        html += f"""
                <div class="card">
                    <h3>{row_name}</h3>
                    <audio controls>
                        <source src="data:audio/flac;base64,{audio_base64}" type="audio/flac">
                        Your browser does not support the audio element.
                    </audio>
                    <a href="data:image/png;base64,{image_base64}" data-lightbox="mel-spectrograms" data-title="{row_name} Mel Spectrogram">
                        <img src="data:image/png;base64,{image_base64}" alt="{row_name} Mel Spectrogram">
                    </a>
                </div>
        """

    html += """
            </div>
        </div>
        <!-- Lightbox2 JS -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox-plus-jquery.min.js"></script>
    </body>
    </html>
    """

    return html


def get_audio_info(file: str) -> tuple:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",  # Select only audio streams
        "-show_entries",
        "stream=channels",  # Get channel count from the stream
        "-show_entries",
        "format=duration",  # Get the duration from the format section
        "-of",
        "json",
        file,
    ]

    # Run ffprobe command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Parse the JSON output
    output = json.loads(result.stdout)

    # Extract channels and duration
    channels = int(output["streams"][0]["channels"])
    duration = float(output["format"]["duration"])

    return channels, duration


def extract_audio_segment(
    file: str, start_time: float, dur: float, sr: int, num_channels: int
) -> np.ndarray:
    # Define the ffmpeg command to output raw PCM data
    command = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        file,
        "-t",
        str(dur),
        "-f",
        "s16le",
        "-ac",
        str(num_channels),
        "-ar",
        str(sr),
        "-loglevel",
        "error",
        "pipe:1",
    ]

    # Run the command and capture the output
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Convert the raw PCM data to a NumPy array
    wav_data = np.frombuffer(process.stdout, dtype=np.int16).copy()

    # Reshape based on the number of channels
    wav_data = wav_data.reshape(-1, num_channels)

    return wav_data


def blur_latent(latent: torch.Tensor, is_augmentation: bool = False):
    def noise_audio(latent, corrupt=0.6):
        c = 1.0 - corrupt
        noised_enc = (latent * c) + torch.randn_like(latent) * (1 - (1 - 1e-4) * c)
        return noised_enc
    
    def shift_tensor(tensor: torch.Tensor, shift: int):
        """
        shift > 0: 뒤로 밀기 (앞에 zero padding 추가)
        shift < 0: 앞으로 밀기 (뒤에 zero padding 추가)
        shift = 0: 변화 없음
        """
        tensor = tensor.unsqueeze(dim=0)
        B, T, D = tensor.shape
        if shift == 0:
            return tensor

        if shift > 0:
            pad = (0, 0, shift, 0)
            padded = F.pad(tensor, pad, mode='constant', value=0)
            return padded[:, :T, :]  # 앞에 pad 추가된 만큼 자름
        else:
            shift = -shift # 뒤에 pad 추가, 앞에서 자름
            pad = (0, 0, 0, shift)
            padded = F.pad(tensor, pad, mode='constant', value=0)
            return padded[:, shift:, :]


    voice_cond = torch.from_numpy(latent.copy()).unsqueeze(dim=0)
    voice_cond = voice_cond.permute(0, 2, 1)            # (B, 64, 48)
    voice_cond = F.avg_pool1d(voice_cond, kernel_size=2, stride=2)  # (B, 64, 24)
    voice_cond = F.interpolate(voice_cond, scale_factor=2, mode='nearest')  # (B, 64, 48)
    voice_cond = voice_cond.permute(0, 2, 1).squeeze()
    voice_cond = noise_audio(voice_cond)

    if is_augmentation:
        shift = random.randint(-3, 3)
        if random.random()<0.5:
            shift = 0
        voice_cond = shift_tensor(voice_cond, shift=shift)
    
    return voice_cond
import torch
import torchaudio
import numpy as np
from scipy.signal import medfilt
from torchaudio.functional import spectral_centroid

N_CHROMA = 24
RADIX2_EXP = 14
WIN_LENGTH = 2 ** RADIX2_EXP
SAMPLE_RATE = 44100
TARGET_FRAMERATE = 21.5
HOP_LENGTH = int(SAMPLE_RATE / TARGET_FRAMERATE)
N_FFT = 2048

# Chroma filterbank
chroma_filterbank = torch.from_numpy(
    filters.chroma(sr=SAMPLE_RATE, n_fft=N_FFT, tuning=0, n_chroma=N_CHROMA)
).float()

# Spectrogram transform
spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    power=2,
    center=True,
    pad=0,
    normalized=True,
)

def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)

def compute_centroid(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    spec = spec_transform(waveform)
    freqs = torch.linspace(0, sample_rate // 2, spec.shape[1], device=spec.device)
    centroid = (spec * freqs[None, :, None]).sum(dim=1) / (spec.sum(dim=1) + 1e-6)
    midi_like = 69 + 12 * torch.log2(centroid / 440.0 + 1e-6)
    return (midi_like / 127.0).clamp(0, 1)

def add_noise(tensor: torch.Tensor, std: float = 0.005) -> torch.Tensor:
    noise = torch.randn_like(tensor) * std
    return (tensor + noise).clamp(0.0, 1.0)  # 정규화 유지

def get_dynamic(waveform: torch.Tensor, max_len: int) -> torch.Tensor:
    if waveform.ndim == 3:
        waveform = waveform.mean(dim=1)
    elif waveform.ndim == 2 and waveform.size(0) == 2:
        waveform = waveform.mean(dim=0)

    spec = spec_transform(waveform)  # [B, F, T]
    chroma = torch.einsum('cf,...ft->...ct', chroma_filterbank, spec)
    chroma_normed = torch.nn.functional.normalize(chroma, p=float('inf'), dim=-2, eps=1e-6)

    # max_indices = chroma_normed.argmax(dim=-2, keepdim=True)  # (B, 1, T)
    # mask = torch.zeros_like(chroma_normed)
    # mask.scatter_(-2, max_indices, 1.0)  # set 1 where chroma bin is max
    # chroma_maxonly = chroma_normed * mask

    chroma_indices = chroma_normed.argmax(dim=-2, keepdim=True)  # (B, 1, T)
    chroma_maxonly = chroma_indices.expand(-1, 4, -1)  # (B, T, 1)

    # mask = (chroma_normed >= 0.8).float()
    # chroma_maxonly = chroma_normed * mask

    # RMS 계산 (PyTorch 버전)
    frame_size = HOP_LENGTH
    waveform_mono = waveform.mean(dim=0) if waveform.dim() == 2 else waveform
    rms_frames = waveform_mono.unfold(0, frame_size, frame_size)
    rms = torch.sqrt((rms_frames ** 2).mean(dim=1))  # [T_rms]
    rms = rms.clamp(min=1e-8)

    # RMS downsample to chroma time resolution
    T_chroma = chroma.size(-1)
    rms_down = torch.nn.functional.interpolate(rms.unsqueeze(0).unsqueeze(0), size=T_chroma, mode='linear', align_corners=False)
    expanded_rms = rms_down.squeeze(0).expand(4, -1).unsqueeze(0)  # [1, 4, T]
    expanded_rms = min_max_normalize(expanded_rms)

    centroid = compute_centroid(waveform, SAMPLE_RATE)[0].unsqueeze(0).expand(4, -1).unsqueeze(0)  # [1, 4, T]

    # add noise
    if random.random() < 0.5:
        expanded_rms = add_noise(expanded_rms)
    if random.random() < 0.5:
        centroid = add_noise(centroid)
    
    combined = torch.cat((centroid, expanded_rms, chroma_maxonly), dim=-2)  # [1, D, T]
    combined = combined.permute(0, 2, 1)  # [1, T, D]

    if combined.shape[1] < max_len:
        pad_len = max_len - combined.shape[1]
        pad = torch.zeros((1, pad_len, combined.shape[2]), device=combined.device)
        combined = torch.cat([combined, pad], dim=1)
    else:
        combined = combined[:, :max_len, :]

    return combined.squeeze()
