# coding:utf-8
"""
Name : malay_audio_dataset.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/24/2021 10:16 AM
Desc:
"""
import os
import sys
import logging
import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
from torch.utils.data import Dataset
from src.dataloader.augment_audio import generate_compose_transform, ComposeTransform

logger = logging.getLogger(__name__)


class MalayAudioDataset(Dataset):
    def __init__(
            self,
            sample_rate=16_000,
            max_sample_size=None,
            min_sample_size=None,
            min_length=0,
            shuffle=True,
            pad=True,
            normalize=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.shuffle = shuffle
        self.pad = pad
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"Sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


class FileMalayAudioDataset(MalayAudioDataset):
    def __init__(
            self,
            annotation_file,
            audio_dir,
            aug_dir=None,
            sample_rate=16_000,
            max_sample_size=None,
            min_sample_size=None,
            shuffle=True,
            min_length=0,
            pad=False,
            normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.aug_dir = aug_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        import soundfile as sf

        audio_file, audio_sample_path = self._get_audio_sample_path(index)
        transcript = self._get_audio_sample_transcript(index)
        wav, curr_sample_rate = sf.read(audio_sample_path)
        audio_transformed = wav
        if self.aug_dir is not None:
            audio_transformed = self._get_augment_audio(wav)
            logger.info(f"Augment {audio_file} into {audio_transformed.shape[0]} waves.")
        return audio_transformed, transcript

    def _get_audio_sample_path(self, index):
        audio_file = f"{self.annotations.iloc[index, 0]}"
        path = os.path.join(self.audio_dir, audio_file)
        return audio_file, path

    def _get_audio_sample_transcript(self, index):
        return self.annotations.iloc[index, 1].strip()

    def _get_augment_audio(self, audio_sample):
        transforms = generate_compose_transform(self.aug_dir)
        compose_transform = ComposeTransform(transforms)
        transformed_audio = compose_transform(audio_sample)
        return torch.from_numpy(np.fromiter(transformed_audio, dtype=np.float32))


if __name__ == "__main__":
    AUDIO_PATH = "D:\Projects\mock_project\wav2vec2-malay\\tests\\waves"
    ANNOTATION_PATH = "D:\Projects\mock_project\wav2vec2-malay\\tests\\annotations.csv"
    AUG_DIR = "D:\Projects\mock_project\wav2vec2-malay\\tests\\aug_dir"
    malay_data = FileMalayAudioDataset(ANNOTATION_PATH, AUDIO_PATH, AUG_DIR)
    print(f"There are {len(malay_data)} samples in the datasets")

    a = 1
