# coding:utf-8
"""
Name : audio_dataloader.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/24/2021 10:16 AM
Desc:
"""
import os

os.environ["WANDB_DISABLED"] = "true"
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from time import time

import datasets
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import load_metric, Dataset
from torch.utils.data import DataLoader
from torchaudio_augmentations import (RandomApply,
                                      PolarityInversion,
                                      Noise,
                                      Gain,
                                      HighLowPass,
                                      Delay,
                                      compose)

# from torchaudio_augmentations.augmentations.pitch_shift import PitchShift
# from torchaudio_augmentations.augmentations.reverb import Reverb
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, \
    TrainingArguments, Trainer

from preprocessing_dataset import remove_special_characters, speech_file_to_array_fn, prepare_dataset
from src.dataloader import AudioDataLoader

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
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
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
            num_augmented_samples=16,
            have_transforms=True,
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
        self.have_transforms = have_transforms
        if self.have_transforms:
            self.transforms = self._get_audio_transforms()
        self.num_augmented_samples = num_augmented_samples

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
        if self.have_transforms:
            transform_many = compose.ComposeMany(transforms=self.transforms,
                                                 num_augmented_samples=self.num_augmented_samples)
            signal = transform_many(signal)
        input_values, labels = self._prepare_dataset(signal, transcript)
        # batch = [{'input_values': input_values, 'labels': labels}]
        # print(index)
        # # input_values = input_values.squeeze().numpy()
        # # signal = signal.to(self.device)
        # data_collator = self._get_data_collator()
        # collate_batch = data_collator(batch)
        return {'input_values': input_values, "labels": labels}

    def _get_audio_sample_path(self, index):
        audio_file = f"{self.annotations.iloc[index, 0]}"
        # print(index, audio_file)
        path = os.path.join(self.audio_dir, audio_file)
        return path

    def _get_audio_sample_transcript(self, index):
        return self.annotations.iloc[index, 1].strip()

    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal

    def _get_audio_transforms(self):
        num_samples = self.sample_rate * 5
        return [
            # RandomResizedCrop(n_samples=num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.1, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.3),
            HighLowPass(sample_rate=self.sample_rate),
            # RandomApply([PitchShift(
            #     n_samples=num_samples,
            #     sample_rate=self.sample_rate
            # )], p=0.4),
            RandomApply([Delay(sample_rate=self.sample_rate)], p=0.5),
            # RandomApply([Reverb(sample_rate=self.sample_rate)], p=0.3)
        ]

    def _prepare_dataset(self, speech_signal, transcript):
        # check that all files have the correct sampling rate
        assert (
                self.sample_rate == 16_000
        ), f"Make sure all inputs have the same sampling rate of {self.audio_processor.feature_extractor.sampling_rate}."

        input_values = self.audio_processor(speech_signal, sampling_rate=self.sample_rate).input_values
        # input_values = input_values[0].squeeze()
        # input_values = torch.from_numpy(input_values)
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


# def collate_fn(batch):
#     # batched = [elem for elem in batch if elem is not None]
#     # if all(batched) is None:
#     #     return
#     input_features, labels = list(zip(*batch))


# def collate_fn(data):
#     """
#        data: is a list of tuples with (example, label, length)
#              where 'example' is a tensor of arbitrary shape
#              and label/length are scalars
#     """
#     labels, lengths = zip(**data)
#     max_len = max(lengths)
#     n_ftrs = data[0][0].size(1)
#     features = torch.zeros((len(data), max_len, n_ftrs))
#     labels = torch.tensor(labels)
#     lengths = torch.tensor(lengths)
#
#     for i in range(len(data)):
#         j, k = data[i][0].size(0), data[i][0].size(1)
#         features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])
#
#     return features.float(), labels.long(), lengths.long()


def parse_args():
    args = argparse.ArgumentParser(description="Data loader for audio")
    args.add_argument("--audio_path", type=str, required=True, help="Path to directory which contains audio")
    args.add_argument("--annotation_path", type=str, required=True, help="Path to csv annotation")
    args.add_argument("--num_augmented_samples", type=int, default=16, required=True,
                      help="Number of augmented samples")
    args.add_argument("--num_workers", type=int, default=1, required=True, help="Number of workers")

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
                                      have_transforms=False,
                                      audio_processor=audio_processor)

    print(f"There are {len(malay_dataset)} samples in the datasets")

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


demo()
