# coding:utf-8
"""
Name : audio_dataloader.py
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
from datasets import load_metric
from torchaudio_augmentations import (RandomApply,
                                      PolarityInversion,
                                      Noise,
                                      Gain,
                                      HighLowPass,
                                      Delay,
                                      compose, PitchShift, Reverb)

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
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.tokenizer = Wav2Vec2CTCTokenizer(self.vocab_path,
                                              unk_token="[UNK]",
                                              pad_token="[PAD]",
                                              word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                          sampling_rate=16000,
                                                          padding_value=0.0,
                                                          do_normalize=True,
                                                          return_attention_mask=True)
        self.audio_processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor,
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
        # input_features = np.column_stack()
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # label_features = collate_fn(label_features)

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
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


class MalayAudioDataset:
    def __init__(
            self,
            annotation_file,
            audio_dir,
            audio_processor,
            sample_rate=16_000,
            audio_transforms=None,
            num_transforms=16,
            padding=True,
            device="cuda"
    ):

        self.audio_processor = audio_processor
        self.tokenizer = audio_processor.tokenizer
        self.feature_extractor = audio_processor.feature_extractor
        self.audio_processor = audio_processor.audio_processor
        self.padding = padding
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.device = device
        self.audio_transforms = audio_transforms
        if self.audio_transforms:
            self._transforms = self._get_audio_transforms()
        self.num_transforms = num_transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        if index >= len(self.annotations):
            raise StopIteration
        if index not in self.annotations.index:
            return
        audio_sample_path = self._get_audio_sample_path(index)
        transcript = self._get_audio_sample_transcript(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        if self._transforms:
            transform_many = compose.ComposeMany(transforms=self._transforms,
                                                 num_augmented_samples=self.num_transforms)
            signal = transform_many(signal)
        input_values, labels = self._prepare_dataset(signal, sr, transcript)

        return {'input_values': input_values, "labels": labels}

    def _get_audio_sample_path(self, index):
        audio_file = f"{self.annotations.iloc[index, 0]}"
        path = os.path.join(self.audio_dir, audio_file)
        return path

    def _get_audio_sample_transcript(self, index):
        return self.annotations.iloc[index, 1].strip()

    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            audio_resample = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = audio_resample(signal)
        return signal

    def _get_audio_transforms(self):
        num_samples = self.sample_rate * random.randint(1, 5)
        return [
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

    def _prepare_dataset(self, speech_signal, sr, transcript):
        # check that all files have the correct sampling rate
        assert (
                sr == 16_000
        ), f"Make sure all inputs have the same sampling rate of {self.audio_processor.feature_extractor.sampling_rate}."

        input_values = self.audio_processor(speech_signal, sampling_rate=self.sample_rate).input_values
        with self.audio_processor.as_target_processor():
            labels = self.audio_processor(transcript).input_ids
        return input_values, labels

    def _get_data_collator(self):
        return DataCollatorCTCWithPadding(processor=self.audio_processor, padding=self.padding)

    def _get_tokenizer(self):
        return self.tokenizer

    def _get_feature_extractor(self):
        return self.feature_extractor

    def _get_processor(self):
        return self.audio_processor


def parse_args():
    args = argparse.ArgumentParser(description="Data loader for audio")
    args.add_argument("--audio_path", type=str, required=True, help="Path to directory which contains audio")
    args.add_argument("--annotation_path", type=str, required=True, help="Path to csv annotation")
    args.add_argument("--num_transform", type=int, default=16, required=True,
                      help="Number of augmented samples")

    return args.parse_args()


def demo():
    time_begin = time()
    AUDIO_PATH = "../tests/waves"
    ANNOTATION_PATH = "../tests/annotations.csv"
    vocab_path = "../tests/vocab.json"

    audio_processor = AudioProcessor(vocab_path=vocab_path)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    malay_dataset = MalayAudioDataset(annotation_file=ANNOTATION_PATH,
                                      audio_dir=AUDIO_PATH,
                                      audio_transforms=True,
                                      audio_processor=audio_processor,
                                      num_transforms=16)

    print(f"There are {len(malay_dataset)} samples in the datasets, "
          f"data augmentation: {malay_dataset.num_transforms}")

    tokenizer = Wav2Vec2CTCTokenizer(vocab_path,
                                     unk_token="[UNK]",
                                     pad_token="[PAD]",
                                     word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)
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
        output_dir="./wav2vec2-base-malay",
        group_by_length=True,
        per_device_train_batch_size=8,
        num_train_epochs=20,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2
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
