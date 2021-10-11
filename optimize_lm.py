# coding:utf-8
"""
Name    : optimize_lm.py
Author  : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 09/10/2021
Desc:
"""
import argparse
import os
import sys
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
Train and optimize a KenLM language model
"""

set_caching_enabled(False)


# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    batch["transcript"] = batch["transcript"].strip()
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
        binarylm_file_path = os.path.join(lm_model_dir, "lm.binary")
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


def train(kenlm_path, output_path, transcript_file_path, corpus_file_path, n_gram=4):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(transcript_file_path, encoding="utf-8") as f:
        train = f.read().lower().splitlines()

    chars = [list(d.replace(' ', '')) for d in train]
    chars = [j for i in chars for j in i]
    chars = set(chars)

    if corpus_file_path is not None:
        with open(corpus_file_path, encoding="utf-8") as f:
            train += f.read().lower().splitlines()

    vocabs = set([])
    for line in tqdm(train):
        for word in line.split():
            vocabs.add(word)
    vocabs = list(vocabs)
    print(len(vocabs))
    vocabs = [v for v in vocabs if not any(c for c in list(v) if c not in chars)]
    print(len(vocabs))

    vocab_path = os.path.join(output_path, 'vocabs.txt')
    lexicon_path = os.path.join(output_path, 'lexicon.txt')
    train_text_path = os.path.join(output_path, 'world_lm_data.train')
    train_text_path_train = os.path.join(output_path, 'kenlm.train')
    model_arpa = os.path.join(output_path, '4gram_big.arpa')
    model_bin = os.path.join(output_path, 'lm.binary')
    kenlm_path_train = os.path.join(abspath(kenlm_path), 'build/bin/lmplz')
    kenlm_path_convert = os.path.join(abspath(kenlm_path), 'build/bin/build_binary')
    kenlm_path_query = os.path.join(abspath(kenlm_path), 'build/bin/query')

    with open(train_text_path, 'w') as f:
        f.write('\n'.join(train))

    with open(vocab_path, 'w') as f:
        f.write(' '.join(vocabs))

    for i in range(0, len(vocabs)):
        vocabs[i] = vocabs[i] + '\t' + ' '.join(list(vocabs[i])) + ' |'

    with open(lexicon_path, 'w') as f:
        f.write('\n'.join(vocabs))

    cmd = kenlm_path_train + " -T /tmp -S 4G --discount_fallback -o " + str(
        n_gram) + " --limit_vocab_file " + vocab_path + " trie < " + train_text_path + ' > ' + model_arpa
    os.system(cmd)
    cmd = kenlm_path_convert + ' trie ' + model_arpa + ' ' + model_bin
    os.system(cmd)
    cmd = kenlm_path_query + ' ' + model_bin + " < " + train_text_path + ' > ' + train_text_path_train
    os.system(cmd)
    os.remove(train_text_path)
    os.remove(train_text_path_train)
    os.remove(model_arpa)
    os.remove(vocab_path)


def optimize(lm_dir,
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
    test_dataset = Dataset.from_pandas(test_df)
    print("Preprocessing speech files")
    test_dataset = test_dataset.map(speech_file_to_array_fn)

    print("Beginning alpha and beta hyperparameter optimization")
    study = optuna.create_study()
    study.optimize(optimize_lm_objective, n_jobs=n_jobs, n_trials=n_trials)

    lm_best = {'alpha': study.best_params['lm_alpha'], 'beta': study.best_params['lm_beta']}

    config_file_path = os.path.join(lm_model_dir, "config_ctc.yaml")
    with open(config_file_path, 'w') as config_file:
        yaml.dump(lm_best, config_file)

    print('Best params saved to config file {}: alpha={}, beta={} with WER={}'.format(config_file_path,
                                                                                      study.best_params['lm_alpha'],
                                                                                      study.best_params['lm_beta'],
                                                                                      study.best_value))


def main(lm_dir,
         test_dataset_path,
         wav2vec2_model_path,
         n_trials,
         n_jobs,
         **args):
    # lm_dir = train(kenlm_path, output_path, transcript_file_path, corpus_file_path, n_gram)
    optimize(lm_dir, test_dataset_path, wav2vec2_model_path, n_trials=n_trials, n_jobs=n_jobs)


def demo():
    lm_dir = "language_model/lm"
    test_dataset_path = "examples/datasets/test.csv"
    wav2vec2_model_path = "model/checkpoint"
    n_trials = 2
    n_jobs = 1
    optimize(lm_dir, test_dataset_path, wav2vec2_model_path, n_trials=n_trials, n_jobs=n_jobs)


if __name__ == "__main__":
    # demo()
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--lm_dir", dest="lm_dir", required=True, help="root directory for language model")
    parser.add_argument("--test_dataset_path", dest="test_dataset_path", required=True,
                        help="directory for test dataset")
    # parser.add_argument("--transcript_file_path", dest="transcript_file_path", required=True, help="transcript "
    #                                                                                                "directory of "
    #                                                                                                "dataset")
    # parser.add_argument("--corpus_file_path", dest="corpus_file_path", required=True,
    #                     help="corpus directory of project")
    # parser.add_argument("--n_gram", default=4, dest="n_gram", required=True, help="n-grams")
    parser.add_argument("--wav2vec2_model_path", dest="wav2vec2_model_path", required=True, help="path to pretrained "
                                                                                                 "model directory")
    parser.add_argument("--n_jobs", default=1, dest="n_jobs", required=False, help="number of jobs executed")
    parser.add_argument("--n_trials", default=10, dest="n_trials", required=False, help="number of trials executed")
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
