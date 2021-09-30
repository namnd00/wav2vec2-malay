# coding:utf-8
"""
Name : preprocessing_dataset.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/26/2021 5:13 PM
Desc:
"""
import re

import numpy as np
import torchaudio
from unidecode import unidecode


def remove_special_characters(batch, chars_to_ignore_regex, pattern_dot_decimal, train=True):
    batch["transcript"] = re.sub(chars_to_ignore_regex, ' ', batch["transcript"]).lower()
    if re.search(pattern_dot_decimal, batch["transcript"]):
        batch["transcript"] = re.sub("&", ' dan ', batch["transcript"])
    else:
        batch["transcript"] = re.sub("&", 'dan', batch["transcript"])

    if train:
        batch["target_text"] += " "
    return batch


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["transcript"]
    return batch


def filter_by_duration(batch):
    return (
            20 >= batch["duration"] >= 1
            and len(batch["target_text"]) > 1
    )


def prepare_dataset(batch, processor):
    # check that all files have the correct sampling rate
    assert (
            batch["sampling_rate"] == 16000
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values
    # Setup the processor for targets
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

# def split_dataset(data_args, annotation_df, prefix="batch_"):
#     # get mask to split dataset to train and test
#     msk = np.random.rand(len(annotation_df)) <= data_args.train_test_split_ratio
#     # get train csv
#     train_csv = annotation_df[msk]
#     # get temp test dataframe to split it to test and val
#     temp_test_csv = annotation_df[~msk]
#     # get mask train to split temp train dataframe to train and val
#     msk_test = np.random.rand(len(temp_test_csv)) <= (1 - data_args.train_test_split_ratio) / 2
#     # split temp train dataframe to train and val
#     test_csv = temp_test_csv[msk_test]
#     eval_csv = temp_test_csv[~msk_test]
#     # save train, val, test to .csv
#     train_path = f'{data_args.dataset_config_name}/{prefix}train.csv'
#     eval_path = f'{data_args.dataset_config_name}/{prefix}eval.csv'
#     test_path = f'{data_args.dataset_config_name}/{prefix}test.csv'
#
#     train_csv.to_csv(train_path, index=False)
#     eval_csv.to_csv(eval_path, index=False)
#     test_csv.to_csv(test_path, index=False)
#
#     return train_path, eval_path, test_path
