# coding:utf-8
"""
Name    : optimize.py
Author  : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 07/10/2021
Desc:
"""

from os.path import abspath
import os
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--kenlm_path", default=None, type=str,
                        required=True, help="Path to kenlm library")

    parser.add_argument("--transcript_file", default=None, type=str, required=True,
                        help="Path to text file")

    parser.add_argument("--additional_file", default=None, type=str, required=False,
                        help="Path to corpus file")

    parser.add_argument("--ngram", default=3, type=int,
                        required=False, help="n-gram")

    parser.add_argument("--output_path", default=None, type=str,
                        required=True, help="Output path for storing pretrained_models")

    args = parser.parse_args()

    return args


def train_lm(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(args.transcript_file, encoding="utf-8") as f:
        train = f.read().lower().splitlines()

    chars = [list(d.replace(' ', '')) for d in train]
    chars = [j for i in chars for j in i]
    chars = set(chars)

    if args.additional_file != None:
        with open(args.additional_file, encoding="utf-8") as f:
            train += f.read().lower().splitlines()

    vocabs = set([])
    for line in tqdm(train):
        for word in line.split():
            vocabs.add(word)
    vocabs = list(vocabs)
    print(len(vocabs))
    vocabs = [v for v in vocabs if not any(c for c in list(v) if c not in chars)]
    print(len(vocabs))

    vocab_path = os.path.join(args.output_path, 'vocabs.txt')
    lexicon_path = os.path.join(args.output_path, 'lexicon.txt')
    corpus_file = os.path.join(args.output_path, 'malay_lm_corpus.txt')
    model_arpa = os.path.join(args.output_path, 'malay_lm.arpa')
    model_bin = os.path.join(args.output_path, 'malay_lm.bin')
    kenlm_path_train = os.path.join(abspath(args.kenlm_path), 'bin/lmplz')
    kenlm_path_convert = os.path.join(abspath(args.kenlm_path), 'bin/build_binary')

    with open(corpus_file, 'w') as f:
        f.write('\n'.join(train))

    with open(vocab_path, 'w') as f:
        f.write(' '.join(vocabs))

    for i in range(0, len(vocabs)):
        vocabs[i] = vocabs[i] + '\t' + ' '.join(list(vocabs[i])) + ' |'

    with open(lexicon_path, 'w') as f:
        f.write('\n'.join(vocabs))

    cmd = "{kenlm_path_train} -o {n} --text {corpus_file} --arpa {lm_file}".format(kenlm_path_train=kenlm_path_train,
                                                                                   n=args.ngram,
                                                                                   corpus_file=corpus_file,
                                                                                   lm_file=model_arpa)
    os.system(cmd)
    cmd = "{kenlm_path_convert} trie {model_arpa} {model_bin}".format(kenlm_path_convert=kenlm_path_convert,
                                                                      model_arpa=model_arpa,
                                                                      model_bin=model_bin)
    os.system(cmd)
    os.remove(corpus_file)
    # os.remove(model_arpa)
    os.remove(vocab_path)

    return model_bin


if __name__ == "__main__":
    args = parse_arguments()
    train_lm(args)
