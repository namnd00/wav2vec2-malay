# coding:utf-8
"""
Name : argument_classes.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/10/2021 9:22 AM
Desc:
"""
from dataclasses import dataclass, field
from typing import List, Optional


def list_field(default=None, metadata=None):
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
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    layerdrop: Optional[float] = field(
        default=0.0, metadata={"help": "The LayerDrop probability."}
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
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_test_split_ratio: Optional[float] = field(
        default=0.8,
        metadata={
            "help": "The ratio of train test split datasets. Defaults to 0.2"
        }
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to "
                    "'submission_2.csv' "
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    chars_to_ignore: List[str] = list_field(
        default=['"', "()", "[\]", "`", "_", "+/=%|"],
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    per_device_test_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for testing."}
    )
    data_augment: bool = field(
        default=False,
        metadata={"help": "Augment speech data or not"}
    )
    ratio_dataset: float = field(
        default=0.8,
        metadata={'help': "ratio data"}
    )
