# coding:utf-8
"""
Name    : optimize.py
Author  : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 09/10/2021
Desc:
"""
import argparse
import os
import sys
from functools import partial

import yaml

import torch
import torchaudio
import optuna
import pandas as pd
from ctcdecode import CTCBeamDecoder
from datasets import Dataset, load_metric, set_caching_enabled
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import abspath

DESCRIPTION = """
Train and optimize a KenLM language pretrained_models
"""

set_caching_enabled(False)


def speech_file_to_array_fn(batch):
    batch["transcript"] = batch["transcript"].strip()  # + " "
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array
    return batch


def decode(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)
    batch["pred_strings_with_lm"] = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]]).strip()

    return batch


def optimize_lm_objective(trial, test_dataset):
    global ctcdecoder

    alpha = trial.suggest_uniform('lm_alpha', 0, 6)
    beta = trial.suggest_uniform('lm_beta', 0, 5)

    try:
        binarylm_file_path = os.path.join(lm_model_dir, "malay_lm.binary")
        ctcdecoder = CTCBeamDecoder(vocab,
                                    model_path=binarylm_file_path,
                                    alpha=alpha,
                                    beta=beta,
                                    cutoff_top_n=40,
                                    cutoff_prob=1.0,
                                    beam_width=100,
                                    num_processes=4,
                                    blank_id=processor.tokenizer.pad_token_id,
                                    log_probs_input=True
                                    )
        result = test_dataset.map(decode)
        result_wer = wer.compute(predictions=result["pred_strings_with_lm"], references=result["sentence"])
        trial.report(result_wer, step=0)

    except Exception as e:
        print(e)
        raise

    finally:
        return result_wer


def optimize(lm_dir,
             dataset_dir,
             test_dataset_path,
             wav2vec_model_path,
             n_trials=5,
             n_jobs=1):
    global processor
    global model
    global vocab
    global wer
    global lm_model_dir

    lm_model_dir = lm_dir

    wer = load_metric("wer")

    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_path)

    model.to("cuda")

    vocab = processor.tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '

    print("Loading dataset")
    test_df = pd.read_csv(test_dataset_path)
    test_df['path'] = dataset_dir + test_df['path']
    test_dataset = Dataset.from_pandas(test_df)
    print("Preprocessing speech files")
    test_dataset = test_dataset.map(speech_file_to_array_fn)

    print("Beginning alpha and beta hyperparameter optimization")
    study = optuna.create_study()
    optimize_lm_func = partial(optimize_lm_objective, test_dataset=test_dataset)
    study.optimize(optimize_lm_func, n_jobs=n_jobs, n_trials=n_trials)

    lm_best = {'alpha': study.best_params['lm_alpha'], 'beta': study.best_params['lm_beta']}

    config_file_path = os.path.join(lm_model_dir, "config_ctc.yaml")
    with open(config_file_path, 'w') as config_file:
        yaml.dump(lm_best, config_file)

    print('Best params saved to config file {}: '
          'alpha={}, beta={} with WER={}'.format(config_file_path,
                                                 study.best_params['lm_alpha'],
                                                 study.best_params['lm_beta'],
                                                 study.best_value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--lm_dir", dest="lm_dir", required=True,
                        help="root directory for language pretrained_models")

    parser.add_argument("--test_dataset_path", dest="test_dataset_path", required=True,
                        help="directory for test dataset")

    parser.add_argument("--wav2vec2_model_path", dest="wav2vec2_model_path", required=True,
                        help="path to pretrained models directory")

    parser.add_argument("--n_jobs", default=1, dest="n_jobs", required=False,
                        help="number of jobs executed")

    parser.add_argument("--n_trials", default=10, dest="n_trials", required=False,
                        help="number of trials executed")

    parser.add_argument("--dataset_dir", dest="dataset_dir", required=True,
                        help="path to test dataset directory")

    parser.set_defaults(func=optimize)
    args = parser.parse_args()
    args.func(**vars(args))
