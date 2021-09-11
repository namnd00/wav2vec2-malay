# coding:utf-8
"""
Name : train.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/9/2021 2:27 PM
Desc:
"""
import re
import time

import pandas as pd
import torchaudio
import wandb
import json
import logging
import os
import sys
import datasets
import numpy as np
import torch
import transformers

import time
import logging
import wandb

from inspect import getframeinfo, stack
from unidecode import unidecode
from datasets import Dataset
from pathlib import Path
# from prepare_dataset import (remove_special_characters,
#                              extract_all_chars,
#                              get_resampler,
#                              speech_file_to_array_fn,
#                              filter_by_duration,
#                              prepare_dataset,
#                              get_length)
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
    EarlyStoppingCallback,
    is_apex_available,
    set_seed,
)

from argument_classes import ModelArguments, DataTrainingArguments
from transforms.spec_augment import Compose, FrequencyMask, TimeMask

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


class Timer:
    """
    Measure time elapsed since start and since last measurement.
    """

    def __init__(self):
        self.start = self.last = time.perf_counter()

    def __call__(self, msg=None):
        t = time.perf_counter()
        timestamp = t - self.start
        duration = t - self.last
        self.last = t
        if msg is None:
            caller = getframeinfo(stack()[1][0])
            msg = f"line {caller.lineno}"
        wandb.log({f"timestamps/{msg}": timestamp})
        wandb.log({f"durations/{msg}": duration})
        logger.info(
            f"*** TIMER *** - {msg} - Timestamp {timestamp:0.4f} seconds - Duration {duration:0.4f} seconds"
        )


class LossNaNStoppingCallback(TrainerCallback):
    """
    Stops training when loss is NaN.

    Loss is accessed through last logged values so it is useful to set
    :class:`~transformers.TrainingArguments` argument `logging_steps` to 1.
    """

    def __init__(self):
        self.stopped = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if np.isnan(logs.get("loss", 0.0)):
            self.stopped = True
            control.should_training_stop = True
            logger.info("Loss NaN detected, terminating training")


class TimingCallback(TrainerCallback):
    """
    Logs some interesting timestamps.
    """

    def __init__(self):
        self.training_started = False
        self.evaluation_started = False

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.training_started:
            log_timestamp("trainer ready to start 1st step")

    def on_step_end(self, args, state, control, **kwargs):
        if not self.training_started:
            self.training_started = True
            log_timestamp("first training step")


logger = logging.getLogger(__name__)
log_timestamp = Timer()


def remove_special_characters(batch, chars_to_ignore_regex, train=True):
    batch["text"] = (
        re.sub(chars_to_ignore_regex, "", unidecode(batch["transcript"]))
            .lower()
            .strip()
    )
    if train:
        batch["text"] += " "
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


resampler = dict()


def get_resampler(sampling_rate):
    if sampling_rate in resampler.keys():
        return resampler[sampling_rate]
    else:
        logger.info(f"Creating new resampler for {sampling_rate}")
        resampler[sampling_rate] = torchaudio.transforms.Resample(
            sampling_rate, 16_000
        )
        return resampler[sampling_rate]


def get_spec_augment(max_width=10, use_mean=False):
    return Compose([
        FrequencyMask(max_width=max_width, use_mean=use_mean),
        TimeMask(max_width=max_width, use_mean=use_mean),
    ])


def get_noise_speech(noise_path=None):
    pass


# Preprocessing the datasets.
# We need to read the audio files as arrays and tokenize the targets.
def speech_file_to_array_fn(batch):
    # if spec_augment:
    #     transforms = get_spec_augment(max_width, use_mean)
    # if add_noise:
    #     pass
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    # if transforms is not None:
    #     speech_array = transforms(speech_array)
    batch["speech"] = get_resampler(sampling_rate)(speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["text"]
    batch["duration"] = len(speech_array.squeeze()) / sampling_rate
    return batch


# def augment_add_noise_speech_file_to_array_fn(batch, augment):
#     speech_array, sampling_rate = torchaudio.load(batch["path"])
#     len_augment = len(augment)
#     id_rand = np.random.randint(int(len_augment * 0.9))
#     speech_array = speech_array + augment[id_rand: len(speech_array) + id_rand] * 0.3
#     batch["speech"] = speech_array
#     batch["sampling_rate"] = sampling_rate
#     batch["transcript"] = batch["transcript"].lower()
#     batch["target_text"] = batch["transcript"]
#     return batch


def filter_by_duration(batch):
    return (
            20 >= batch["duration"] >= 1
            and len(batch["target_text"]) >= 1
    )  # about 98% of samples


def prepare_dataset(batch, processor):
    # check that all files have the correct sampling rate
    assert (
            len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values
    # Setup the processor for targets
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


def get_length(item):
    # speeds up grouping by length in pre-loaded dataset
    item["length"] = len(item["input_values"])
    return item


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

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

    # override default run name and log all args
    wandb.init(project="wav2vec2-malay", config=wandb.config)

    # Detecting last checkpoint.
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

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)]
                        )
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

    chars_to_ignore_regex = f'[{"".join(data_args.chars_to_ignore)}]'

    # Pre-processed datasets
    # dataset_path = Path(os.getenv("HF_HOME", ".")) / "datasets"
    # train: datasets/train
    # eval: datasets/eval
    # test: datasets/test
    # vocab: datasets/vocab.json
    dataset_train_path = f"{data_args.dataset_config_name}/train"
    dataset_eval_path = f"{data_args.dataset_config_name}/eval"
    dataset_test_path = f"{data_args.dataset_config_name}/test"

    vocab_path = f"{data_args.dataset_config_name}/vocab.json"

    train_csv = None
    eval_csv = None
    test_csv = None
    eval_dataset = None

    if Path(data_args.train_split_name).exists():
        sub_df = pd.read_csv(data_args.train_split_name, encoding='utf-8')
        msk = np.random.rand(len(sub_df)) <= data_args.train_test_split_ratio
        temp_csv = sub_df[msk]
        msk_train = np.random.rand(len(temp_csv)) <= data_args.train_test_split_ratio
        train_csv = temp_csv[msk_train]
        eval_csv = temp_csv[~msk_train]
        test_csv = sub_df[~msk]
        train_csv.to_csv(f'{data_args.dataset_config_name}/train.csv')
        eval_csv.to_csv(f'{data_args.dataset_config_name}/eval.csv')
        test_csv.to_csv(f'{data_args.dataset_config_name}/test.csv')
        logger.info(f"Train size: {len(train_csv)} - Val size: {len(eval_csv)} - Test size: {len(test_csv)}")
        logger.info(f"split data to train/test->"
                    f"{data_args.train_test_split_ratio}/{1 - data_args.train_test_split_ratio}")

    if Path(dataset_train_path).exists() and Path(vocab_path).exists():
        train_dataset = datasets.load_from_disk(dataset_train_path)
        log_timestamp("load pre-processed data")
    else:
        train_dataset = Dataset.from_pandas(train_csv)
        log_timestamp("load train data")
        train_dataset = train_dataset.map(
            lambda x: remove_special_characters(x, chars_to_ignore_regex, train=False),
            remove_columns=["transcript"]
        )
        log_timestamp("remove special characters")

    if training_args.do_eval:
        if Path(dataset_eval_path).exists():
            eval_dataset = datasets.load_from_disk(dataset_eval_path)
            log_timestamp("load pre-processed data")
        else:
            eval_dataset = Dataset.from_pandas(eval_csv)
            log_timestamp("load val data")
            eval_dataset = eval_dataset.map(
                lambda x: remove_special_characters(x, chars_to_ignore_regex, train=False),
                remove_columns=["transcript"]
            )
            log_timestamp("remove special characters")

    if Path(dataset_test_path).exists() and Path(vocab_path).exists():
        test_dataset = datasets.load_from_disk(dataset_test_path)
        log_timestamp("load pre-processed data")
    else:
        test_dataset = Dataset.from_pandas(test_csv)
        log_timestamp("load test data")
        test_dataset = test_dataset.map(
            lambda x: remove_special_characters(x, chars_to_ignore_regex, train=False),
            remove_columns=["transcript"],
        )
        log_timestamp("remove special characters")

    if not Path(vocab_path).exists():
        # create vocab
        vocab_train = train_dataset.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=train_dataset.column_names,
        )
        vocab_test = test_dataset.map(
            extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=test_dataset.column_names,
        )
        vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
        log_timestamp("create vocab")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
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
    log_timestamp("load model")

    if not Path(dataset_train_path).exists():
        train_dataset = train_dataset.map(
            speech_file_to_array_fn,
            remove_columns=train_dataset.column_names,
            num_proc=data_args.preprocessing_num_workers,
        )
        log_timestamp("load audio")
        train_dataset = train_dataset.filter(
            filter_by_duration,
            remove_columns=["duration"],
            num_proc=data_args.preprocessing_num_workers,
        )
        log_timestamp("filter data")
        train_dataset = train_dataset.map(
            lambda x: prepare_dataset(x, processor),
            remove_columns=train_dataset.column_names,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        log_timestamp("process data")
        train_dataset = train_dataset.map(
            get_length,
            num_proc=data_args.preprocessing_num_workers,
        )
        log_timestamp("add input length")
        train_dataset.save_to_disk(dataset_train_path)
        log_timestamp("save to disk")

    if not Path(dataset_eval_path).exists() and training_args.do_eval:
        eval_dataset = eval_dataset.map(
            speech_file_to_array_fn,
            remove_columns=eval_dataset.column_names,
            num_proc=data_args.preprocessing_num_workers,
        )
        eval_dataset = eval_dataset.filter(
            filter_by_duration,
            remove_columns=["duration"],
            num_proc=data_args.preprocessing_num_workers,
        )
        eval_dataset = eval_dataset.map(
            lambda x: prepare_dataset(x, processor),
            remove_columns=eval_dataset.column_names,
            batch_size=training_args.per_device_eval_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
        eval_dataset = eval_dataset.map(
            get_length,
            num_proc=data_args.preprocessing_num_workers,
        )
        eval_dataset.save_to_disk(dataset_eval_path)

    if not Path(dataset_test_path).exists():
        test_dataset = test_dataset.map(
            speech_file_to_array_fn,
            num_proc=data_args.preprocessing_num_workers,
        )
        test_dataset = test_dataset.filter(
            filter_by_duration, remove_columns=["duration"]
        )
        test_dataset.save_to_disk(dataset_test_path)

    # Metric
    cer_metric = datasets.load_metric("cer")
    # we use a custom WER that considers punctuation
    wer_metric = datasets.load_metric("wer")

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

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()
        log_timestamp("freeze feature extractor")

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    log_timestamp("create data collator")

    # Initialize our Trainer
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )
    loss_nan_stopping_callback = LossNaNStoppingCallback()
    timing_callback = TimingCallback()
    trainer.add_callback(loss_nan_stopping_callback)
    trainer.add_callback(timing_callback)

    # Training
    log_timestamp("setup trainer")
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        log_timestamp()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        log_timestamp("train model")
        trainer.save_model()

        # save the feature_extractor and the tokenizer
        if is_main_process(training_args.local_rank):
            processor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Final test metrics
    logger.info("*** Test ***")
    log_timestamp()

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
            predictions=result["pred_strings"], references=result["text"]
        )
        test_wer = wer_metric.compute(
            predictions=result["pred_strings"], references=result["text"]
        )
        log_timestamp("compute test metrics")

    metrics = {"cer": test_cer, "wer": test_wer}
    wandb.log({f"test/{k}": v for k, v in metrics.items()})
    trainer.save_metrics("test", metrics)
    logger.info(metrics)

    # save model files
    log_timestamp()
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
