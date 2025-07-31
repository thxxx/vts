import json
from typing import Any, Literal, TypeVar

import torch
from einops import rearrange, reduce
from huggingface_hub.file_download import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F

from vocos.dac.model import DAC
from vocos.dac.vae import DACVAE
from vocos.feature_extractors import FeatureExtractor
from vocos.oobleck import AutoencoderOobleck
from vocos.pretrained import Vocos
from vocos.utils import mask_from_lengths

REPO_NAME = "optimizerai/vocos"


def get_voco(voco_type: str):
    match voco_type:
        case "mel":
            voco = MelVoco()
        case "encodec":
            voco = EncodecVoco()
        case "dac":
            voco = DACVoco()
        case "dacvae":
            voco = DACVAEVoco()
        case "oobleck":
            voco = OobleckVoco()
        case _:
            raise ValueError(f"Unsupported voco: {voco_type}")

    voco.eval()
    for param in voco.parameters():
        param.requires_grad_(False)

    return voco


def load(name: str):
    config_path = hf_hub_download(repo_id=REPO_NAME, filename=f"{name}.json")
    model_path = hf_hub_download(repo_id=REPO_NAME, filename=f"{name}.safetensors")
    with open(config_path, "r") as f:
        config: dict[str, Any] = json.load(f)

    state_dict = load_file(model_path, device="cpu")

    return config, state_dict


Length = TypeVar("Length", int, Tensor)


class Voco(nn.Module):
    feature_extractor: nn.Module
    channel: Literal[1, 2]

    @property
    def latent_dim(self) -> int:
        raise NotImplementedError()

    @property
    def sampling_rate(self) -> int:
        raise NotImplementedError()

    @torch.no_grad()
    def encode(self, audio: Tensor) -> Tensor:
        mel = self.feature_extractor(audio)
        mel = rearrange(mel, "b d n -> b n d")
        return mel

    def decode(self, latents: Tensor) -> Tensor:
        raise NotImplementedError()

    def encode_length(self, lengths: Length) -> Length:
        raise NotImplementedError()

    def decode_length(self, lengths: Length) -> Length:
        raise NotImplementedError()

    def encode_mask(self, mask: Tensor) -> Tensor:
        length = reduce(mask, "b l -> b", "sum")
        max_length = mask.shape[1]
        conv_length = self.encode_length(length)
        max_conv_length = self.encode_length(max_length)
        return mask_from_lengths(conv_length, max_conv_length)

    def decode_mask(self, mask: Tensor) -> Tensor:
        length = reduce(mask, "b l -> b", "sum")
        max_length = mask.shape[1]
        conv_length = self.decode_length(length)
        max_conv_length = self.decode_length(max_length)
        return mask_from_lengths(conv_length, max_conv_length)


class MelVoco(Voco):
    def __init__(self):
        super().__init__()
        config, state_dict = load("mel")
        model = Vocos.from_config(**config)
        model.load_state_dict(state_dict)
        model.eval()
        self.head = model.head
        self.feature_extractor = model.feature_extractor
        self.backbone = model.backbone
        self.channel = 1

    @property
    def latent_dim(self) -> int:
        return self.feature_extractor.mel_spec.n_mels

    @property
    def sampling_rate(self) -> int:
        return self.feature_extractor.mel_spec.sample_rate

    def decode(self, latents: Tensor) -> Tensor:
        latents = rearrange(latents.float(), "b n d -> b d n")
        audio = self.head(self.backbone(latents))
        return rearrange(audio, "b n -> b n ()")

    def encode_length(self, lengths: Length) -> Length:
        return lengths // self.feature_extractor.mel_spec.hop_length + 1

    def decode_length(self, lengths: Length) -> Length:
        return lengths * self.feature_extractor.mel_spec.hop_length - 1


class EncodecVoco(Voco):
    def __init__(self, bandwidth_id=2):
        super().__init__()
        config, state_dict = load("encodec")
        model = Vocos.from_config(**config)
        model.load_state_dict(state_dict)
        model.eval()
        self.feature_extractor = model.feature_extractor
        self.backbone = model.backbone
        self.head = model.head
        self.register_buffer("bandwidth_id", torch.tensor([bandwidth_id]))
        self.bandwidth_id: Tensor
        self.feature_extractor.set_target_bandwidth(bandwidth_id)
        self.channel = 1

    @property
    def latent_dim(self) -> int:
        return self.feature_extractor.encodec.encoder.dimension

    @property
    def compression_factor(self) -> int:
        bandwidth = int(self.feature_extractor.encodec.bandwidth * 1000)
        num_quantizers = (
            self.feature_extractor.encodec.quantizer.get_num_quantizers_for_bandwidth(
                self.feature_extractor.encodec.frame_rate,
                self.feature_extractor.encodec.bandwidth,
            )
        )
        bits_per_codebook = self.feature_extractor.encodec.bits_per_codebook
        codec_rate = bandwidth // num_quantizers // bits_per_codebook
        return self.sampling_rate // codec_rate

    @property
    def sampling_rate(self) -> int:
        return self.feature_extractor.encodec.sample_rate

    def decode(self, latents: Tensor) -> Tensor:
        latents = rearrange(latents, "b n d -> b d n")
        audio = self.head(self.backbone(latents, bandwidth_id=self.bandwidth_id))
        return rearrange(audio, "b n -> b n ()")

    def encode_length(self, lengths: Length) -> Length:
        return (lengths - 1) // self.compression_factor + 1

    def decode_length(self, lengths: Length) -> Length:
        return lengths * self.compression_factor


class DACFeatures(FeatureExtractor):
    def __init__(self, dac: DAC | DACVAE, channel: Literal[1, 2]):
        super().__init__()
        self.encoder = dac.encoder
        self.hop_length = int(dac.hop_length)
        self.channel = channel

    def preprocess(self, audio: Tensor):
        length = audio.shape[-1]
        right_pad = ((length - 1) // self.hop_length + 1) * self.hop_length - length
        audio = F.pad(audio, (0, right_pad))
        return audio

    def forward(self, audio: Tensor):
        audio = rearrange(audio, "b t c -> (b c) () t")
        audio = self.preprocess(audio)
        if self.channel == 1:
            latent = self.encoder(audio)
        else:
            latent, _ = self.encoder(audio)
        latent = rearrange(latent, "(b c) d t -> b (c d) t", c=self.channel)
        return latent


class DACVoco(Voco):
    def __init__(self):
        super().__init__()
        config, state_dict = load("dac")
        for key_g in list(state_dict.keys()):
            if "weight_g" in key_g:
                key_v = key_g.replace("weight_g", "weight_v")
                key = key_g.replace(".weight_g", ".weight")
                state_dict[key] = torch._weight_norm(
                    state_dict[key_v], state_dict[key_g]
                )
                del state_dict[key_g]
                del state_dict[key_v]

        dac = DAC(**config)
        dac.load_state_dict(state_dict)
        dac.eval()
        self.feature_extractor = DACFeatures(dac, 1)
        self.decoder = dac.decoder
        self.hop_length = int(dac.hop_length)
        self._latent_dim = dac.latent_dim
        self._sampling_rate = dac.sample_rate
        self.channel = 1

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    def decode(self, latents: Tensor) -> Tensor:
        latents = rearrange(latents, "b n (c d) -> (b c) d n", c=self.channel)
        audio = self.decoder(latents)
        audio = rearrange(audio, "(b c) () t -> b t c", c=self.channel)
        return audio

    def encode_length(self, lengths: Length) -> Length:
        return (lengths - 1) // self.hop_length + 1

    def decode_length(self, lengths: Length) -> Length:
        return lengths * self.hop_length


class DACVAEVoco(Voco):
    def __init__(self, model_type="small"):
        super().__init__()
        match model_type:
            case "small":
                config, state_dict = load("dac-vae-small")
            case _:
                raise NotImplementedError()

        for key_g in list(state_dict.keys()):
            if "original0" in key_g:
                key_v = key_g.replace("original0", "original1")
                key = key_g.replace(".original0", "").replace(".parametrizations", "")
                state_dict[key] = torch._weight_norm(
                    state_dict[key_v], state_dict[key_g]
                )
                del state_dict[key_g]
                del state_dict[key_v]

        dac_vae = DACVAE(**config)
        dac_vae.load_state_dict(state_dict)
        dac_vae.eval()
        self.channel = 2
        self.feature_extractor = DACFeatures(dac_vae, self.channel)
        self.decoder = dac_vae.decoder
        self.hop_length = int(dac_vae.hop_length)
        self._latent_dim = dac_vae.latent_dim
        self._sampling_rate = dac_vae.sample_rate

    @property
    def latent_dim(self) -> int:
        return self._latent_dim * self.channel

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        latents = rearrange(latents, "b n (c d) -> (b c) d n", c=self.channel)
        audio = self.decoder(latents)
        audio = rearrange(audio, "(b c) () t -> b t c", c=self.channel)
        return audio

    def encode_length(self, lengths: Length) -> Length:
        return (lengths - 1) // self.hop_length + 1

    def decode_length(self, lengths: Length) -> Length:
        return lengths * self.hop_length


class OobleckFeatures(FeatureExtractor):
    def __init__(self, oobleck: AutoencoderOobleck):
        super().__init__()
        self.encoder = oobleck.encoder
        self.hop_length = int(oobleck.hop_length)

    def preprocess(self, audio: Tensor):
        length = audio.shape[-1]
        right_pad = ((length - 1) // self.hop_length + 1) * self.hop_length - length
        audio = F.pad(audio, (0, right_pad))
        return audio

    def forward(self, audio: Tensor):
        audio = rearrange(audio, "b t c -> b c t")
        audio = self.preprocess(audio)
        latent = self.encoder(audio)
        mean, scale = latent.chunk(2, dim=1)
        std = nn.functional.softplus(scale) + 1e-4
        sample = torch.randn_like(std)
        x = mean + std * sample
        return x


class OobleckVoco(Voco):
    def __init__(self):
        super().__init__()
        config, state_dict = load("oobleck")

        for key_g in list(state_dict.keys()):
            if "weight_g" in key_g:
                key_v = key_g.replace("weight_g", "weight_v")
                key = key_g.replace(".weight_g", ".weight")
                state_dict[key] = torch._weight_norm(
                    state_dict[key_v], state_dict[key_g]
                )
                del state_dict[key_g]
                del state_dict[key_v]

        model = AutoencoderOobleck(**config)
        model.load_state_dict(state_dict)
        self.feature_extractor = OobleckFeatures(model)

        self.decoder = model.decoder
        self.hop_length = int(model.hop_length)
        self._latent_dim = model.encoder_hidden_size
        self._sampling_rate = model.sampling_rate
        self.channel = 2

    @property
    def latent_dim(self) -> int:
        return self._latent_dim // 2

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        latents = rearrange(latents, "b n d -> b d n")
        audio = self.decoder(latents)
        audio = rearrange(audio, "b c t -> b t c")
        return audio

    def encode_length(self, lengths: Length) -> Length:
        return (lengths - 1) // self.hop_length + 1

    def decode_length(self, lengths: Length) -> Length:
        return lengths * self.hop_length
