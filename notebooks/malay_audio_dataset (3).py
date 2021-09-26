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

import numpy as np
import pandas as pd
import soundfile
import torch
import torchaudio
from torchvision import transforms
from audiomentations.core.audio_loading_utils import load_sound_file
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import augment_audio
from augment_audio import get_transforms

logger = logging.getLogger(__name__)


class MalayAudioDataset(Dataset):
    def __init__(
            self,
            annotation_file,
            audio_dir,
            sample_rate=16_000,
            transforms=None,
            device="cpu"
    ):
        super().__init__()
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.target_sample_rate = 16000
        self.device = device
       
        self.transforms = transforms
    
    def __len__(self):
        if self.transforms is not None:
            return len(self.transforms) * len(self.annotations)
        else:
            return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        transcript = self._get_audio_sample_transcript(index)
        signal, sr = load_sound_file(audio_sample_path, sample_rate=self.sample_rate)
        signal = self._resample_if_necessary(signal, sr)
        #print("Before: ", signal.shape)
        if self.transforms is not None:
            for aug in self.transforms:
             
              aug_signal = aug(signal, self.target_sample_rate)
              signal_out = np.vstack((signal, aug_signal))
        #print("After: ", signal.shape)
        signal = torch.from_numpy(signal_out)
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


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    AUDIO_PATH = "/content/drive/MyDrive/wav2vec2-malay/tests/waves"
    ANNOTATION_PATH = "/content/drive/MyDrive/wav2vec2-malay/tests/annotations.csv"
    AUG_DIR = "/content/drive/MyDrive/wav2vec2-malay/tests/aug_dir"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    class_transforms = get_transforms(aug_dir=AUG_DIR)

    malay_data = MalayAudioDataset(audio_dir=AUDIO_PATH, annotation_file=ANNOTATION_PATH,transforms = class_transforms)
    loader = DataLoader(dataset=malay_data, batch_size=4, num_workers=1, collate_fn=collate_fn)

    print(f"There are {len(malay_data)} samples in the datasets")
    print(len(loader))
    
    for i in tqdm(range(len(loader))):
      batch, label = next(iter(loader))

