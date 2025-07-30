from pathlib import Path
from typing import Literal
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split

from data.dataset_video import AudioDataset


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        # dataset_paths: list[Path],
        batch_size: int,
        num_workers: int,
        max_audio_len: int,
        sampling_rate: int,
        channel: Literal[1, 2],
        max_txt_len: int,
        pin_memory: bool = False,
        drop_last: bool = True,
        max_video_len: int = 31
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dataset_path = dataset_path
        self.max_video_len = max_video_len
        # self.dataset_paths = [path.expanduser() for path in dataset_paths]

        self.max_audio_len = max_audio_len
        self.sampling_rate = sampling_rate
        self.channel: Literal[1, 2] = channel
        self.max_txt_len = max_txt_len

        self.dataset: AudioDataset | None = None

    def setup(self, stage: str | None = None):
        if self.dataset is None:
            self.dataset = AudioDataset(
                dataset_path=self.dataset_path,
                max_audio_len=self.max_audio_len,
                max_txt_len=self.max_txt_len,
                sampling_rate=self.sampling_rate,
                max_video_len=self.max_video_len,
                channel=self.channel,
            )
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [0.95, 0.04, 0.01],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
            collate_fn=self.dataset.collate,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
            collate_fn=self.dataset.collate,
            prefetch_factor=4,
        )

    def test_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
            collate_fn=self.dataset.collate,
            prefetch_factor=4,
        )

    def predict_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
            collate_fn=self.dataset.collate,
            prefetch_factor=4,
        )
