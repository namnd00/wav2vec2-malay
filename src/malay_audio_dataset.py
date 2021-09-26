# coding:utf-8
"""
Name : malay_audio_dataset.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/24/2021 10:16 AM
Desc:
"""
import os
import logging
import argparse

import pandas as pd
import torch
import torchaudio
from torchaudio_augmentations import *
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class MalayAudioDataset(Dataset):
    def __init__(
            self,
            annotation_file,
            audio_dir,
            sample_rate=16_000,
            num_augmented_samples=16,
            have_transforms=True,
            device="cuda"
    ):
        super().__init__()
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.device = device
        self.have_transforms = have_transforms
        if self.have_transforms:
            self.transforms = self._get_audio_transforms()
        self.num_augmented_samples = num_augmented_samples

    def __len__(self):
        if self.have_transforms:
            return len(self.transforms) * len(self.annotations)
        else:
            return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        transcript = self._get_audio_sample_transcript(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        if self.have_transforms:
            transform_many = compose.ComposeMany(transforms=self.transforms,
                                                 num_augmented_samples=self.num_augmented_samples)
            signal = transform_many(signal)

        signal = torch.from_numpy(signal)
        signal = signal.to(self.device)
        return signal, transcript

    def _get_audio_sample_path(self, index):
        audio_file = f"{self.annotations.iloc[index, 0]}"
        path = os.path.join(self.audio_dir, audio_file)
        return path

    def _get_audio_sample_transcript(self, index):
        return self.annotations.iloc[index, 1].strip()

    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal

    # def _filter_by_duration(self, index):
    #     return 1 <= self.annotations.iloc[index, 2] <= 30

    def _get_audio_transforms(self):
        num_samples = self.sample_rate * 5
        return [
            RandomResizedCrop(n_samples=num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.1, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.3),
            HighLowPass(sample_rate=self.sample_rate),
            RandomApply([PitchShift(
                n_samples=num_samples,
                sample_rate=self.sample_rate
            )], p=0.4),
            RandomApply([Delay(sample_rate=self.sample_rate)], p=0.5),
            RandomApply([Reverb(sample_rate=self.sample_rate)], p=0.3)
        ]


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args():
    args = argparse.ArgumentParser()
    args.description("Data loader for audio")
    args.add_argument("--audio_path", type=str, required=True, help="Path to directory which contains audio")
    args.add_argument("--annotation_path", type=str, required=True, help="Path to csv annotation")
    args.add_argument("--num_augmented_samples", type=int, default=16, required=True, help="Number of augmented samples")

    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    AUDIO_PATH = args['audio_path']
    ANNOTATION_PATH = args['annotation_path']
    num_augmented_samples = args['num_augmented_samples']

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    malay_data = MalayAudioDataset(audio_dir=AUDIO_PATH,
                                   annotation_file=ANNOTATION_PATH,
                                   num_augmented_samples=num_augmented_samples)
    loader = DataLoader(dataset=malay_data,
                        batch_size=8,
                        num_workers=2,
                        collate_fn=collate_fn)

    print(f"There are {len(malay_data)} samples in the datasets")
    print(len(loader))
    for batch, label in loader:
        print(type(batch))
        print(type(batch[0]))
    a = 1
