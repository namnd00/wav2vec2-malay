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
import io
import sys
import json
import yaml
import shlex
import torch
import torchaudio
import optuna
import pandas as pd

from functools import partial
from pathlib import Path
from ctcdecode import CTCBeamDecoder
from datasets import Dataset, load_metric, set_caching_enabled
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from argparse import ArgumentParser, RawTextHelpFormatter

DESCRIPTION = """
Train and optimize a KenLM language model
"""

set_caching_enabled(False)


# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array
    return batch


def optimize_lm_objective(trial,
                          lm_model_dir,
                          processor,
                          test_dataset,
                          wer,
                          vocab,
                          model):
    def decode(batch):
        inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

        beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)
        batch["pred_strings_with_lm"] = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]]).strip()

        return batch

    alpha = trial.suggest_uniform('lm_alpha', 0, 6)
    beta = trial.suggest_uniform('lm_beta', 0, 5)

    try:
        binarylm_file_path = os.path.join(lm_model_dir, "malay_lm.bin")
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


def optimize(lm_model_dir,
             test_dataset_path,
             dataset_dir,
             wav2vec_model_path,
             n_trials,
             n_jobs):
    test_dataset = pd.read_csv(test_dataset_path)
    test_dataset['path'] = dataset_dir + "/" + test_dataset['path']
    test_dataset = Dataset.from_pandas(test_dataset)

    print("Loading wer")
    wer = load_metric("wer")

    print("Loading processor and pretrained model")
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_path)

    model.to("cuda")

    print("Create vocab")
    vocab = processor.tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '
    print(f"Vocab size: {len(vocab)}")
    print(vocab)

    print("Preprocessing speech files")
    test_dataset = test_dataset.map(speech_file_to_array_fn)

    print("Beginning alpha and beta hyperparameter optimization")
    study = optuna.create_study()
    print("Call optimize lm func")
    optimize_lm_func = partial(optimize_lm_objective,
                               lm_model_dir=lm_model_dir,
                               processor=processor,
                               test_dataset=test_dataset,
                               wer=wer,
                               vocab=vocab,
                               model=model)
    print("Begin optimizing...")
    study.optimize(optimize_lm_func, n_jobs=n_jobs, n_trials=n_trials)

    print("End optimizing.")
    lm_best = {'alpha': study.best_params['lm_alpha'], 'beta': study.best_params['lm_beta']}

    print("Dump lm best params to config ctc")
    config_file_path = os.path.join(lm_model_dir, "config_ctc.yaml")
    with open(config_file_path, 'w') as config_file:
        yaml.dump(lm_best, config_file)

    print('Best params saved to config file {}: alpha={}, beta={} with WER={}'.format(config_file_path,
                                                                                      study.best_params['lm_alpha'],
                                                                                      study.best_params['lm_beta'],
                                                                                      study.best_value))


def parse_arguments():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--lm_dir", required=True,
                        help="root directory for language pretrained_models")

    parser.add_argument("--test_dataset_path", required=True,
                        help="directory for test dataset")

    parser.add_argument("--wav2vec2_model_path", required=True,
                        help="path to pretrained models directory")

    parser.add_argument("--n_jobs", type=int, default=2, required=False,
                        help="number of jobs executed")

    parser.add_argument("--n_trials", type=int, default=10, required=False,
                        help="number of trials executed")

    parser.add_argument("--dataset_dir", required=True,
                        help="path to test dataset directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    lm_model_dir = args.lm_dir
    test_dataset_path = args.test_dataset_path
    dataset_dir = args.dataset_dir
    wav2vec_model_path = args.wav2vec2_model_path
    n_trials = args.n_trials
    n_jobs = args.n_jobs

    optimize(lm_model_dir,
             test_dataset_path,
             dataset_dir,
             wav2vec_model_path,
             n_trials=n_trials,
             n_jobs=n_jobs)
