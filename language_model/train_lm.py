# coding:utf-8
"""
Name    : optimize_lm.py
Author  : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 07/10/2021
Desc:
"""

from os.path import abspath
import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--kenlm_path", default=None, type=str,
                        required=True, help="Path to kenlm library")

    parser.add_argument("--transcript_file", default=None, type=str, required=True,
                        help="Path to text file")

    parser.add_argument("--additional_file", default=None, type=str, required=False,
                        help="Path to text file")

    parser.add_argument("--n_gram", default=3, type=int,
                        required=False, help="Ngram")

    parser.add_argument("--output_path", default=None, type=str,
                        required=True, help="Output path for storing model")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(args.transcript_file, encoding="utf-8") as f:
        train = f.read().lower().splitlines()

    chars = [list(d.replace(' ', '')) for d in train]
    chars = [j for i in chars for j in i]
    chars = set(chars)

    if args.additional_file is not None:
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
    train_text_path = os.path.join(args.output_path, 'world_lm_data.train')
    train_text_path_train = os.path.join(args.output_path, 'kenlm.train')
    model_arpa = os.path.join(args.output_path, '4gram_big.arpa')
    model_bin  = os.path.join(args.output_path, 'lm.bin')
    kenlm_path_train = os.path.join(abspath(args.kenlm_path), 'build/bin/lmplz')
    kenlm_path_convert = os.path.join(abspath(args.kenlm_path), 'build/bin/build_binary')
    kenlm_path_query = os.path.join(abspath(args.kenlm_path), 'build/bin/query')

    with open(train_text_path, 'w') as f:
        f.write('\n'.join(train))

    with open(vocab_path, 'w') as f:
        f.write(' '.join(vocabs))

    for i in range(0, len(vocabs)):
        vocabs[i] = vocabs[i] + '\t' + ' '.join(list(vocabs[i])) + ' |'

    with open(lexicon_path, 'w') as f:
        f.write('\n'.join(vocabs))

    cmd = kenlm_path_train + " -T /tmp -S 4G --discount_fallback -o " + str(
        args.ngram) + " --limit_vocab_file " + vocab_path + " trie < " + train_text_path + ' > ' + model_arpa
    os.system(cmd)
    cmd = kenlm_path_convert + ' trie ' + model_arpa + ' ' + model_bin
    os.system(cmd)
    cmd = kenlm_path_query + ' ' + model_bin + " < " + train_text_path + ' > ' + train_text_path_train
    os.system(cmd)
    os.remove(train_text_path)
    os.remove(train_text_path_train)
    os.remove(model_arpa)
    os.remove(vocab_path)


if __name__ == "__main__":
    main()
