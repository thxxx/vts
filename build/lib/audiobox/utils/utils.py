import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor

from utils.mask import mask_from_lengths
from utils.typing import EncTensor, LengthTensor

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
