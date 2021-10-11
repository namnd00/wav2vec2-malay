# coding:utf-8
"""
Name : argument_classes.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/10/2021 9:22 AM
Desc:
"""
import argparse
from dataclasses import dataclass, field
from typing import List, Optional


def list_field(default=None, metadata=None):
    """
    @param default:
    @param metadata:
    @return:
    """
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to freeze the feature extractor layers of the model."
        },
    )
    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    hidden_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."
        },
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Propability of each feature vector along the time axis to be chosen as the start of the vector"
                    "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
                    "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is "
                    "True``. "
        },
    )
    layerdrop: Optional[float] = field(
        default=0.1, metadata={"help": "The LayerDrop probability."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_config_name: Optional[str] = field(
        default="./datasets",
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_dir: Optional[str] = field(
        default="./datasets/waves",
        metadata={
            "help": "The path to directory contains audio wav"
        },
    )

    train_data_csv: Optional[str] = field(
        default="./datasets/train.csv",
        metadata={
            "help": "The path of the train dataset csv"
        },
    )
    eval_data_csv: Optional[str] = field(
        default="./datasets/eval.csv",
        metadata={
            "help": "The path of the eval dataset csv"
        },
    )
    test_data_csv: Optional[str] = field(
        default="./datasets/test.csv",
        metadata={
            "help": "The path of the test dataset csv"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    
    transform: Optional[bool] = field(
        default=False, metadata={'help': "Transform audio?"}
    )


@dataclass
class ParameterArguments:
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={"help": "The	 learning rate"},
    )
    weight_decay: Optional[float] = field(
        default=0.005,
        metadata={"help": "The weight decay"},
    )
    warmup_steps: Optional[float] = field(
        default=1000,
        metadata={"help": "The warmup steps"},
    )
    gradient_accumulation_steps: Optional[float] = field(
        default=1,
        metadata={"help": "The gradient accumulation steps"},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine_with_restarts",
        metadata={"help": "The learning rate scheduler type"},
    )

