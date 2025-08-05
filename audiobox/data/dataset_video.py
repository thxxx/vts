import csv
import logging
import random
from pathlib import Path
from typing import Literal
import torchaudio
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import re
import torch.nn.functional as F
from einops import rearrange

def get_npy_filename(path: str, duration: float) -> str:
    base_name = '.'.join(path.split('.')[:-1])
    max_index = int(duration / 18.604)

    # rand_index = random.randint(0, max_index)
    rand_index = 0
    return f"{base_name}.{rand_index:04d}.npy"

def build_new_npy_path(orig_audio_path: str):
    audio_wavpath = orig_audio_path.replace("/data/", "/")
    path_parts = audio_wavpath.split(os.sep)
    new_dir = os.path.join(os.sep, path_parts[1], path_parts[2],  "dynamic_context", *path_parts[3:-1])
    base_name = os.path.splitext(path_parts[-1])[0] + ".npy"
    return os.path.join(new_dir, base_name)

class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        max_audio_len: int,
        max_txt_len: int,
        sampling_rate: int = 44100,
        max_video_len: int = 64,
        channel: Literal[1, 2] = 2,
    ):
        super().__init__()
        self.logger = logging.getLogger("dataset")
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        try:
            output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            output_dir = "outputs"

        if os.path.exists(f"{output_dir}/dataset.log"):
            handler = logging.FileHandler(f"{output_dir}/dataset.log")
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # for logging
        self.max_txt_len = max_txt_len

        self.paths = []
        with open(dataset_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.paths.append((row["audio_path"], row['video_latent_path'], row["caption"], row['duration'], row['sync_latent_path']))

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.tokenizer.padding_side = "right"

        self.sampling_rate = sampling_rate
        self.channel = channel
        self.rng = np.random.default_rng() # np.random이랑 같은데 좀 더 업그레이드된 버전

        self.max_latent_len = max_audio_len
        self.max_video_len = max_video_len

    def __getitem__(self, index: int):
        audio_path, video_path, desc, duration, sync_latent_path = self.paths[index]

        path = audio_path
        # path = get_npy_filename(audio_path, float(duration))
        if not os.path.exists(path):
            print("%s does not exist.", path)
            self.logger.error("%s does not exist.", path)
            return self.__getitem__(index + 1)
        else:
            latent = np.load(path, mmap_mode="r") # N, D
            latent_len = len(latent)
        
        # C, T, W, H
        sync_latent = np.load(sync_latent_path, mmap_mode='r')
        sync_latent = torch.from_numpy(sync_latent.copy())
        sync_latent = rearrange(sync_latent, 'l d -> 1 d l')
        sync_latent = F.interpolate(sync_latent, size=latent_len, mode='nearest-exact').squeeze()
        sync_latent = rearrange(sync_latent, 'd l -> l d')
        
        # compile을  안쓴다면 같은 배치 안의 최대 길이로 하는게 낫긴하지?
        if latent_len < self.max_latent_len:
            latent = np.pad(latent, ((0, self.max_latent_len - latent_len), (0, 0)))
            sync_latent = F.pad(sync_latent, (0, 0, 0, self.max_latent_len - latent_len))
        elif self.max_latent_len < latent_len:
            random_start = self.rng.integers(latent_len - self.max_latent_len)
            latent = latent[random_start : random_start + self.max_latent_len]
            sync_latent = sync_latent[random_start : random_start + self.max_latent_len]
            latent_len = self.max_latent_len
        
        # C, T, W, H
        video_latent = torch.load(video_path)
        # video_latent = np.load(video_path, mmap_mode='r')
        video_latent = video_latent[:8*8, :]
        video_len = video_latent.shape[0]
        
        if video_len < self.max_video_len:
            pad_len = self.max_video_len - video_len
            # video_latent = np.pad(video_latent, ((0,0), (0, pad_len), (0,0), (0,0)))
            video_latent = F.pad(video_latent, (0, 0, 0, 0, 0, pad_len))  # (W, H, T 방향으로 패딩)
        elif self.max_video_len < video_len:
            video_latent = video_latent[:self.max_video_len, :]
            video_len = self.max_video_len

        
        # 원래 latent 길이 ~ max_latent 길이까지는 패딩임을 알려준다.
        audio_mask = torch.arange(self.max_latent_len) < latent_len
        latents = torch.from_numpy(latent.copy())
        # video_latents = torch.from_numpy(video_latent.copy())

        return latents, video_latent, audio_mask, desc, duration, video_path, sync_latent

    def __len__(self):
        return len(self.paths)

    def collate(
        self, batches: list[tuple[Tensor, Tensor, str] | None]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = len(batches)
        filter_batches = [batch for batch in batches if batch is not None]
        if not filter_batches:
            raise ValueError(
                "Something has gravely gone wrong; no valid batch detected."
            )
        while len(filter_batches) < batch_size:
            random_indice = self.rng.integers(len(self))
            if (batch := self[random_indice]) is not None:
                self.logger.info("Adding %d", random_indice)
                filter_batches.append(batch)
            else:
                self.logger.info("%d is a dud", random_indice)

        audio_embed, video_latents, audio_masks, descs, durations, video_path, sync_latent = zip(*filter_batches)
        # audio_embed, audio_masks, descs = zip(*filter_batches)

        batch_encoding = self.tokenizer(
            [desc + self.tokenizer.eos_token for desc in descs],
            return_tensors="pt",
            truncation="longest_first",
            padding="max_length",
            max_length=self.max_txt_len,
            add_special_tokens=False,
        )
        input_ids = batch_encoding.input_ids
        attention_mask = batch_encoding.attention_mask > 0

        # print('durations : ', max(durations)) # 대충 9.9초?
        return torch.stack(audio_embed), torch.stack(video_latents), torch.stack(sync_latent), torch.stack(audio_masks), input_ids, attention_mask, video_path
