import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio import transforms as T

from vocos.encodec.model import EncodecModel
from vocos.modules import safe_log


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: Tensor) -> Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio):
        audio = rearrange(audio, "b n () -> b n")
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = F.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: list[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        self.encodec.eval()
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat(
            [vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]],
            dim=0,
        )
        self.codebook_weights = nn.Parameter(
            codebook_weights, requires_grad=train_codebooks
        )
        self.bandwidths = bandwidths

    def set_target_bandwidth(self, bandwidth_id: int):
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])

    def forward(self, audio: Tensor):
        audio = rearrange(audio, "b n () -> b () n")
        emb = self.encodec.encoder(audio)
        return emb
