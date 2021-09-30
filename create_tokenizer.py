# coding:utf-8
"""
Name : create_tokenizer.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/27/2021 8:38 AM
Desc:
"""
from pathlib import Path
import pandas as pd
from typing import Dict
import re
import itertools
import argparse
import json
import time
import datasets

chars_to_ignore_regex = ['"', "'", "*", "()", "[\]", "\-", "`", "_", "+/=%|"]
pattern_dot_decimal = "\S+\&\S+"
CHARS_TO_IGNORE = f'[{"".join(chars_to_ignore_regex)}]'


def remove_special_characters(batch):
    batch["transcript"] = re.sub(CHARS_TO_IGNORE, ' ', batch["transcript"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcript"])
    vocab = list(set(all_text))
    return {"vocab": vocab, "all_text": [all_text]}


def create_tokenizer(dataframe: pd.DataFrame) -> Dict:
    """
    :param dataframe:
    :return:
    """
    data = datasets.Dataset.from_pandas(dataframe)
    print("remove special characters ......")
    data = data.map(remove_special_characters)
    print("extract all chars ..........")
    vocabs = data.map(extract_all_chars)
    chain = itertools.chain.from_iterable(vocabs['vocab'])
    time.sleep(2)
    vocab_list = list(set(list(chain)))
    return {v: k for k, v in enumerate(vocab_list)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create tokenizer wav2vec2.')
    parser.add_argument('--data_csv',
                        type=str,
                        required=True,
                        help="str - csv directory path : type (path_file, transcripts)")
    parser.add_argument('--path_json_output',
                        type=str,
                        default='vocab.json',
                        required=False,
                        help="str - json directory path")

    args = parser.parse_args()
    df = pd.read_csv(args.data_csv, encoding='utf-8')
    vocab_dict = create_tokenizer(df)
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    Path(args.path_json_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.path_json_output, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
