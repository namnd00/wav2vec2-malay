# coding:utf-8
"""
Name : audio_dataset.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/24/2021 10:16 AM
Desc:
"""
import os
import logging
import argparse
import torch
import torchaudio
from torch.utils.data import Dataset
from dataclasses import dataclass

from torchaudio_augmentations import (RandomApply,
                                      PolarityInversion,
                                      Noise,
                                      Gain,
                                      HighLowPass,
                                      Delay,
                                      Reverb,
                                      Compose)

from typing import (Dict,
                    List,
                    Optional,
                    Union)
from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor)

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(
            self,
            vocab_path,
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
    ):
        self.vocab_path = vocab_path
        self.tokenizer = Wav2Vec2CTCTokenizer(self.vocab_path,
                                              unk_token=unk_token,
                                              pad_token=pad_token,
                                              word_delimiter_token=word_delimiter_token,
                                              )
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=feature_size,
                                                          sampling_rate=sampling_rate,
                                                          padding_value=padding_value,
                                                          do_normalize=do_normalize,
                                                          return_attention_mask=return_attention_mask)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor,
                                           tokenizer=self.tokenizer)

    def __call__(self):
        return self


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"][0].squeeze()} for feature in features
        ]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        del features

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


class MalayAudioDataset(Dataset):
    def __init__(
            self,
            annotation_df,
            audio_dir,
            audio_processor,
            sample_rate=16_000,
            audio_transforms=False,
            device="cuda",
            dataset='train',
    ):
        self.audio_processor = audio_processor.processor
        self.annotation_df = annotation_df
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.device = device
        self.audio_transforms = audio_transforms
        self.dataset = dataset
        if self.dataset not in ['train', 'eval']:
            raise Exception

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        labels = self._get_labels(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        if self.audio_transforms and self.dataset == 'train':
            _transforms = _get_audio_transforms(sr)
            transform = Compose(transforms=_transforms)
            signal = transform(signal)
        signal = self._prepare_signal(signal, sr)
        labels = self._prepare_label(labels)

        return {'input_values': signal, "labels": labels}

    def _get_audio_sample_path(self, index):
        audio_file = f"{self.annotation_df.iloc[index, 0]}"
        path = os.path.join(self.audio_dir, audio_file)
        return path

    def _get_labels(self, index):
        return self.annotation_df.iloc[index, 1].strip().lower()

    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            audio_resample = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = audio_resample(signal)
        return signal

    def _prepare_signal(self, speech_signal, sr):
        # check that all files have the correct sampling rate
        assert (
                sr == 16_000
        ), f"Make sure all inputs have the same sampling rate of {self.audio_processor.feature_extractor.sampling_rate}."

        input_values = self.audio_processor(speech_signal, sampling_rate=self.sample_rate).input_values
        return input_values

    def _prepare_label(self, transcript):
        with self.audio_processor.as_target_processor():
            labels = self.audio_processor(transcript).input_ids
        return labels


def _get_audio_transforms(sr):
    return [
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise(min_snr=0.1, max_snr=0.5)], p=0.3),
        RandomApply([Gain()], p=0.3),
        HighLowPass(sample_rate=sr),
        RandomApply([Delay(sample_rate=sr)], p=0.5),
        RandomApply([Reverb(sample_rate=sr)], p=0.3)
    ]


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["transcript"]
    return batch


def parse_args():
    args = argparse.ArgumentParser(description="Data loader for audio")
    args.add_argument("--audio_path", type=str, required=True, help="Path to directory which contains audio")
    args.add_argument("--annotation_path", type=str, required=True, help="Path to csv annotation")
    args.add_argument("--num_transform", type=int, default=16, required=True,
                      help="Number of augmented samples")

    return args.parse_args()
