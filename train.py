# coding:utf-8
"""
Name : train.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/9/2021 2:27 PM
Desc:
"""
import pandas as pd
import os
import sys

import torchaudio
from torch.cuda import amp

import datasets
import numpy as np
import torch
import transformers

import logging
import wandb

from audio_dataset import MalayAudioDataset, AudioProcessor, DataCollatorCTCWithPadding
from callbacks import TimingCallback
from datasets import Dataset
from pathlib import Path
from typing import Any, Dict, Union
from packaging import version
from torch import nn
from transformers.trainer_utils import (get_last_checkpoint,
                                        is_main_process)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    set_seed,
)

from argument_classes import ModelArguments, DataTrainingArguments
from utils.timer import Timer

logger = logging.getLogger(__name__)
log_timestamp = Timer()

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


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


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["transcript"]
    return batch


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
    VOCAB_PATH = f"{data_args.dataset_config_name}/vocab.json"

    # override default run name and log all args
    wandb.init(project="wav2vec2-malay", config=wandb.config)

    # Detecting last checkpoint.
    last_checkpoint = detect_and_get_last_checkpoint(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
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

    if not Path(VOCAB_PATH).exists():
        logger.error("Must import vocab")

    dataset_train_df = pd.read_csv(data_args.train_data_csv)
    dataset_eval_df = pd.read_csv(data_args.eval_data_csv)
    dataset_test_df = pd.read_csv(data_args.test_data_csv)

    audio_processor = AudioProcessor(vocab_path=VOCAB_PATH)
    audio_dir = data_args.dataset_dir
    train_dataset = MalayAudioDataset(annotation_df=dataset_train_df,
                                      audio_dir=audio_dir,
                                      audio_transforms=data_args.transform,
                                      audio_processor=audio_processor,
                                      dataset='train')
    log_timestamp("Load train dataset")

    eval_dataset = MalayAudioDataset(annotation_df=dataset_eval_df,
                                     audio_dir=audio_dir,
                                     audio_processor=audio_processor,
                                     dataset='eval')
    log_timestamp("Load eval dataset")

    n_train = len(dataset_train_df)
    n_eval = len(dataset_eval_df)
    n_test = len(dataset_test_df)
    logger.info(f"- Train size: {n_train} "
                f"- Val size: {n_eval} "
                f"- Test size: {n_test}")

    # Data collator
    processor = audio_processor.processor
    processor.tokenizer.save_pretrained(training_args.output_dir)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    log_timestamp("create data collator")
    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    log_timestamp("load model")
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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )

    timing_callback = TimingCallback()
    trainer.add_callback(timing_callback)
    log_timestamp("setup trainer")
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

    # Final test metrics
    logger.info("*** Test ***")
    test_dataset = Dataset.from_pandas(dataset_test_df)
    test_dataset = test_dataset.map(
        lambda x: speech_file_to_array_fn(x),
        batch_size=training_args.per_device_eval_batch_size,
        num_proc=1
    )
    log_timestamp("Load test dataset")

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
        evaluate, batch_size=training_args.per_device_eval_batch_size
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
