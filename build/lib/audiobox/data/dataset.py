import csv
import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list[Path],
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

        self.max_txt_len = max_txt_len

        self.dataset_paths = dataset_paths
        self.paths: list[tuple[int, str, str]] = []
        self.augment_dict: dict[int, tuple[str, ...]] = {}
        for i, dataset_path in enumerate(dataset_paths):
            dataset_path = dataset_path.expanduser()
            csv_path = dataset_path / "data.csv"
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.paths.append((i, row["path"], row["desc"]))
                    if "augment_9" in row and row["augment_9"]:
                        self.augment_dict[len(self.paths) - 1] = tuple(
                            row[f"augment_{i}"] for i in range(10)
                        ) + (row["caption"],)

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.tokenizer.padding_side = "right"

        self.sampling_rate = sampling_rate
        self.channel = channel
        self.rng = np.random.default_rng()

        self.max_latent_len = max_audio_len

    def __getitem__(self, index: int):
        dataset_index, subpath, desc = self.paths[index]
        dataset_path = self.dataset_paths[dataset_index]
        path = dataset_path / subpath
        if not path.exists():
            self.logger.error("%s does not exist.", path)
            return None

        if index in self.augment_dict:
            desc_list = self.augment_dict[index] + (desc,)
            desc = random.choice(desc_list)

        latent = np.load(path, mmap_mode="r")
        latent_len = len(latent)

        if latent_len < 4:
            self.logger.error(f"{path} too short: {latent_len} frames.")
            return None

        if latent_len < self.max_latent_len:
            latent = np.pad(latent, ((0, self.max_latent_len - latent_len), (0, 0)))
        elif self.max_latent_len < latent_len:
            random_start = self.rng.integers(latent_len - self.max_latent_len)
            latent = latent[random_start : random_start + self.max_latent_len]
            latent_len = self.max_latent_len
        audio_mask = torch.arange(self.max_latent_len) < latent_len

        return torch.from_numpy(latent.copy()), audio_mask, desc

    def __len__(self):
        return len(self.paths)

    def collate(
        self, batches: list[tuple[Tensor, Tensor, str] | None]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

        audio_embed, audio_masks, descs = zip(*filter_batches)

        batch_encoding = self.tokenizer(
            [desc + self.tokenizer.eos_token for desc in descs],
            return_tensors="pt",
            truncation="longest_first",
            padding="max_length",
            max_length=self.max_txt_len,
            add_special_tokens=False,
        )
        tokens = batch_encoding.input_ids
        token_mask = batch_encoding.attention_mask > 0

        return torch.stack(audio_embed), torch.stack(audio_masks), tokens, token_mask
