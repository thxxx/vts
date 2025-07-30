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
from utils.utils import get_dynamic
import re
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
        sampling_rate: int,
        max_txt_len: int,
        channel: Literal[1, 2],
    ):
        super().__init__()
        self.logger = logging.getLogger("dataset")
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        try:
            output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            output_dir = "outputs"
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
                # self.paths.append((row["audio_path"], row["fined_caption"], row['duration']))
                self.paths.append((row["audio_path"], row["fined_caption"], row['duration'], row['style'], row['loudness_binary'], row['pitch_range']))

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.tokenizer.padding_side = "right"

        self.sampling_rate = sampling_rate
        self.channel = channel
        self.rng = np.random.default_rng() # np.random이랑 같은데 좀 더 업그레이드된 버전

        self.max_latent_len = max_audio_len

    def add_desc(self, path, caption, duration, style, loudness, pitch):
        new_caption = caption

        # start_at = int(path.split(".")[-2])*18.604
        # new_caption += f' & total duration: {duration}s, start at: {start_at}s'
        if loudness is not None and not loudness == '':
            if random.random()<0.5:
                new_caption += f' & {loudness} loudness'

        if pitch in ['low', 'rich', 'high']:
            if random.random()<0.5:
                new_caption += f' & {pitch} pitch range'

        if style is not None and not style == '':
            if random.random()<0.5:
                new_caption += f' in {style} style.'
        
        return new_caption.lower()

    def __getitem__(self, index: int):
        audio_path, desc, duration, style, loudness, pitch = self.paths[index]
        # audio_path, desc, duration = self.paths[index]
        audio_wavpath = re.sub("/data/", "/", audio_path)
        audio_dynamic_path = build_new_npy_path(audio_path)

        waveform, sr = torchaudio.load(audio_wavpath)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        waveform = waveform.mean(dim=0, keepdim=True)

        dynamic_context = get_dynamic(waveform, self.max_latent_len)

        desc = self.add_desc(audio_path, desc, duration, style, loudness, pitch)
        path = get_npy_filename(audio_path, float(duration))

        if not os.path.exists(path):
            print("%s does not exist.", path)
            self.logger.error("%s does not exist.", path)
            return self.__getitem__(index + 1)

        latent = np.load(path, mmap_mode="r")
        latent_len = len(latent)

        if latent_len < 4:
            self.logger.error(f"{path} too short: {latent_len} frames.")
            return self.__getitem__(index + 1)

        if latent_len < self.max_latent_len:
            latent = np.pad(latent, ((0, self.max_latent_len - latent_len), (0, 0)))
        elif self.max_latent_len < latent_len:
            random_start = self.rng.integers(latent_len - self.max_latent_len)
            latent = latent[random_start : random_start + self.max_latent_len]
            latent_len = self.max_latent_len
        
        audio_mask = torch.arange(self.max_latent_len) < latent_len

        latents = torch.from_numpy(latent.copy())

        return latents, audio_mask, desc, dynamic_context, duration
        # return latents, audio_mask, desc

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

        audio_embed, audio_masks, descs, voice_cond, durations = zip(*filter_batches)
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

        # return torch.stack(audio_embed), torch.stack(audio_masks), input_ids, attention_mask
        return torch.stack(audio_embed), torch.stack(audio_masks), input_ids, attention_mask, torch.stack(voice_cond)
