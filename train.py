# coding:utf-8
"""
Name : train.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/9/2021 2:27 PM
Desc:
"""
import re
import augment
import pandas as pd
import torchaudio
import json
import os
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import numpy as np
import torch
import transformers

import time
import logging
import wandb

from inspect import getframeinfo, stack
from unidecode import unidecode

from callbacks import LossNaNStoppingCallback, TimingCallback
from datasets import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from packaging import version
from torch import nn
from transformers.trainer_utils import (get_last_checkpoint,
                                        is_main_process)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainerCallback,
    is_apex_available,
    set_seed,
)

from argument_classes import ModelArguments, DataTrainingArguments
from preprocessing_dataset import split_dataset, create_tokenizer, remove_special_characters, prepare_dataset, \
    speech_file_to_array_fn
from src.audio_dataloader import MalayAudioDataset, collate_fn
from utils import Timer

logger = logging.getLogger(__name__)
log_timestamp = Timer()

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


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
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

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


class CTCTrainer(Trainer):
    def training_step(self,
                      model: nn.Module,
                      inputs: Dict[str, Union[torch.Tensor, Any]]
                      ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(
                    f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']"
                )

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def get_parser():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args


def detect_and_get_last_checkpoint(training_args):
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def main():
    # get args
    model_args, data_args, training_args = get_parser()
    # CONSTANTS
    CHARS_TO_IGNORE = f'[{"".join(data_args.chars_to_ignore)}]'
    VOCAB_PATH = f"{data_args.dataset_config_name}/vocab.json"

    # override default run name and log all args
    wandb.init(project="wav2vec2-malay", config=wandb.config)

    # Detecting last checkpoint.
    last_checkpoint = detect_and_get_last_checkpoint(training_args)

    # Setup logging
    format_basic_config = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    datefmt_basic_config = "%m/%d/%Y %H:%M:%S"
    handlers_basic_config = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(filename="basic_config.json",
                        format=format_basic_config,
                        datefmt=datefmt_basic_config,
                        handlers=handlers_basic_config)
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not Path(data_args.train_split_name).exists():
        logger.error("Must import dataset")

    annotation_df = pd.read_csv(data_args.train_split_name, encoding='utf-8')
    dataset_train_path, dataset_eval_path, dataset_test_path = split_dataset(data_args, annotation_df)

    train_dataset = MalayAudioDataset(annotation_file=dataset_train_path,
                                           audio_dir=data_args.audio_dir,
                                           sample_rate=16000,
                                           num_augmented_samples=data_args.num_augmented_samples,
                                           have_transforms=True,
                                           device=training_args.device)

    log_timestamp("Generate dataset")

    n_samples = len(annotation_df)
    n_train = len(train_dataset)
    n_eval = n_test = (n_samples - n_train) / 2
    logger.info(f"- Train size: {n_train} "
                f"- Val size: {n_eval} "
                f"- Test size: {n_test}")
    logger.info(f"Split data to train/val/test->"
                f"{n_train / n_samples * 100:02}"
                f"/{n_eval / n_samples * 100:02}"
                f"/{n_test / n_samples * 100:02}")

    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = Dataset.from_pandas(dataset_eval_path)
        log_timestamp("load val data")
        eval_dataset = eval_dataset.map(
            lambda x: remove_special_characters(x, CHARS_TO_IGNORE, train=False),
            num_proc=data_args.preprocessing_num_workers
        )
        log_timestamp("Eval: remove special characters")

        eval_dataset = eval_dataset.map(
            speech_file_to_array_fn,
            remove_columns=eval_dataset.column_names,
            num_proc=data_args.preprocessing_num_workers,
        )
        log_timestamp("Eval: speech to array")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_PATH,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        gradient_checkpointing=model_args.gradient_checkpointing,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()
        log_timestamp("freeze feature extractor")

    log_timestamp("load model")

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    log_timestamp("create data collator")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        # train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )
    loss_nan_stopping_callback = LossNaNStoppingCallback()
    timing_callback = TimingCallback()
    trainer.add_callback(loss_nan_stopping_callback)
    trainer.add_callback(timing_callback)
    log_timestamp("setup trainer")

    # Training
    # train_dataloader = trainer.get_train_dataloader()
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=training_args.per_device_train_batch_size,
                                  num_workers=data_args.num_workers,
                                  collate_fn=collate_fn,
                                  shuffle=True)

    log_timestamp("Create train data loader")

    iter_loader = iter(train_dataloader)
    for _ in tqdm(range(len(train_dataloader))):
        train_batch = next(iter_loader)
        log_timestamp("Train: Load audio array, transcripts")

        train_batch = Dataset.from_dict(train_batch)
        train_batch = train_batch.map(
            lambda x: remove_special_characters(x, CHARS_TO_IGNORE, train=False)
        )
        log_timestamp("Train: remove special characters in transcripts")

        train_batched = train_batch.map(
            lambda x: prepare_dataset(x, processor),
            remove_columns=train_batch.column_names,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        log_timestamp("Train: prepare speech array")
        trainer.train_dataset = train_batched

        if training_args.do_train:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_args.model_name_or_path):
                checkpoint = model_args.model_name_or_path
            else:
                checkpoint = None

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            log_timestamp("train model")
            trainer.save_model()

            # save the feature_extractor and the tokenizer
            if is_main_process(training_args.local_rank):
                processor.save_pretrained(training_args.output_dir)

            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            wandb.log({f"train/{k}": v for k, v in metrics.items()})

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    # Prepare test dataset
    test_dataset = Dataset.from_pandas(dataset_test_path)
    log_timestamp("load test data")
    test_dataset = test_dataset.map(
        lambda x: remove_special_characters(x, CHARS_TO_IGNORE, train=False),
        remove_columns=["transcript"],
    )
    log_timestamp("Test: remove special characters")

    test_dataset_dir = f"{data_args.dataset_config_name}/test"
    if not Path(test_dataset_dir).exists():
        test_dataset = test_dataset.map(
            speech_file_to_array_fn,
            num_proc=data_args.preprocessing_num_workers,
        )
        test_dataset.save_to_disk(test_dataset_dir)

    # Metric
    cer_metric = datasets.load_metric("cer")
    # we use a custom WER that considers punctuation
    wer_metric = datasets.load_metric("wer")

    # Final test metrics
    logger.info("*** Test ***")
    result_dict = None
    if loss_nan_stopping_callback.stopped:
        test_cer, test_wer = 1.0, 2.0
        logger.info(
            "Loss NaN detected, typically resulting in bad WER & CER so we won't calculate them."
        )
    else:

        def evaluate(batch):
            inputs = processor(
                batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                logits = model(
                    inputs.input_values.to("cuda"),
                    attention_mask=inputs.attention_mask.to("cuda"),
                ).logits
            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_strings"] = processor.batch_decode(pred_ids)
            return batch

        model.to("cuda")
        # no need to cache mapped test_dataset
        datasets.set_caching_enabled(False)
        result = test_dataset.map(
            evaluate, batched=True, batch_size=training_args.per_device_eval_batch_size
        )
        log_timestamp("get test predictions")
        test_cer = cer_metric.compute(
            predictions=result["pred_strings"], references=result["target_text"]
        )
        test_wer = wer_metric.compute(
            predictions=result["pred_strings"], references=result["target_text"]
        )
        log_timestamp("compute test metrics")

        result_dict = {'labels': test_dataset['target_text'], 'predictions': result['pred_strings']}
    if result_dict is not None:
        result_df = pd.DataFrame(data=result_dict)
        result_df.to_csv(f"{data_args.dataset_config_name}/result_df.csv", index=False)

    metrics = {"cer": test_cer, "wer": test_wer}
    wandb.log({f"test/{k}": v for k, v in metrics.items()})
    trainer.save_metrics("test", metrics)
    logger.info(metrics)

    # save model files
    if not loss_nan_stopping_callback.stopped:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", type="model", metadata={"cer": test_cer}
        )
        for f in Path(training_args.output_dir).iterdir():
            if f.is_file():
                artifact.add_file(str(f))
        wandb.run.log_artifact(artifact)
        log_timestamp("log artifacts")


if __name__ == "__main__":
    main()
