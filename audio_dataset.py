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
import random
from dataclasses import dataclass
from time import time

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from datasets import load_metric
from torchaudio_augmentations import (RandomApply,
                                      PolarityInversion,
                                      Noise,
                                      Gain,
                                      HighLowPass,
                                      Delay,
                                      compose, PitchShift, Reverb, Compose)

from typing import (Dict,
                    List,
                    Optional,
                    Union)
from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor,
                          Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer)

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
            padding=True,
            device="cuda",
            dataset='train',
    ):
        self.audio_processor = audio_processor.processor
        self.padding = padding
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
        if index >= len(self.annotation_df):
            raise StopIteration
        if index not in self.annotation_df.index:
            return
        audio_sample_path = self._get_audio_sample_path(index)
        labels = self._get_labels(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        if self.audio_transforms and self.dataset == 'train':
            _transforms = self._get_audio_transforms(sr)
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

    @staticmethod
    def _get_audio_transforms(sr):
        num_samples = sr * random.randint(1, 5)
        return [
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.1, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.3),
            HighLowPass(sample_rate=sr),
            RandomApply([PitchShift(
                n_samples=num_samples,
                sample_rate=sr
            )], p=0.4),
            RandomApply([Delay(sample_rate=sr)], p=0.5),
            RandomApply([Reverb(sample_rate=sr)], p=0.3)
        ]

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


def parse_args():
    args = argparse.ArgumentParser(description="Data loader for audio")
    args.add_argument("--audio_path", type=str, required=True, help="Path to directory which contains audio")
    args.add_argument("--annotation_path", type=str, required=True, help="Path to csv annotation")
    args.add_argument("--num_transform", type=int, default=16, required=True,
                      help="Number of augmented samples")

    return args.parse_args()


def demo():
    time_begin = time()
    AUDIO_PATH = "/home/namndd3/Documents/wav2vec2-malay/examples/datasets/waves"
    ANNOTATION_PATH = "/home/namndd3/Documents/wav2vec2-malay/examples/datasets/annotations.csv"
    vocab_path = "/home/namndd3/Documents/wav2vec2-malay/examples/datasets/vocab.json"

    audio_processor = AudioProcessor(vocab_path=vocab_path)
    ANNOTATION_DF = pd.read_csv(ANNOTATION_PATH)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    malay_dataset = MalayAudioDataset(annotation_df=ANNOTATION_DF,
                                      audio_dir=AUDIO_PATH,
                                      audio_transforms=True,
                                      audio_processor=audio_processor,
                                      dataset='train')

    print(
        f"There are {len(malay_dataset)} samples in the datasets, data augmentation: {malay_dataset.audio_transforms}")

    # dataloader = DataLoader(dataset=malay_dataset, batch_size=8)

    processor = audio_processor.processor
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    print("Loading models")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    training_args = TrainingArguments(
        output_dir="utils/wav2vec2-base-malay",
        group_by_length=False,
        per_device_train_batch_size=8,
        num_train_epochs=20,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        report_to=['wandb']
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=malay_dataset,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    time_end = time()
    print(f"{time_end - time_begin:02}")
    a = 1


# demo()
