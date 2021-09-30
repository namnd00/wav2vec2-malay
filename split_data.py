import argparse
import os
from pathlib import Path

import pandas as pd
import hashlib

import torchaudio
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset.')
    parser.add_argument('--dataset_dir',
                        type=str,
                        required=True,
                        help="str - dataset directory path")
    parser.add_argument('--data_csv',
                        type=str,
                        required=True,
                        help="str - csv directory path")
    parser.add_argument('--split_batch',
                        default=True,
                        type=bool,
                        required=True,
                        help="bool - split dataset to multiple batch?")
    parser.add_argument('--n_batch',
                        default=3,
                        type=int,
                        required=False,
                        help="int - number of batch")
    parser.add_argument('--n_split',
                        default=3,
                        type=int,
                        required=True,
                        help="int - number of data to split to train/test or train/eval/test")
    parser.add_argument('--train_ratio',
                        default=0.9,
                        type=float,
                        required=True,
                        help="int - ratio of data to split to train/test or train/eval/test")
    parser.add_argument('--wav_dir',
                        type=str,
                        required=True,
                        help="str - path to directory contain waves")
    parser.add_argument('--text_dir',
                        type=str,
                        required=True,
                        help="str - path to directory contain text")
    parser.add_argument('--prefix_batch',
                        type=str,
                        default='train_batch_',
                        required=False,
                        help="str - prefix of file batch, i.e: batch_0.csv")
    parser.add_argument('--min_duration',
                        type=int,
                        default=1,
                        required=False,
                        help="int - min duration")
    parser.add_argument('--max_duration',
                        type=int,
                        default=20,
                        required=False,
                        help="int - max duration")
    parser.add_argument('--rename',
                        type=bool,
                        required=True,
                        help="bool - rename files in dataset dir?")
    parser.add_argument('--refine_data',
                        type=bool,
                        default=True,
                        required=True,
                        help="bool - refine data by duration of audio?")
    parser.add_argument('--refined_data_csv',
                        type=str,
                        required=True,
                        help="str - refine dataset csv directory path")

    return parser.parse_args()


def calc_checksum(f):
    return str(hashlib.sha3_512(f.encode('utf-8')).hexdigest()[0:8])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def rename_files_and_get_annotations(args):
    text_dir = args.text_dir
    wav_dir = args.wav_dir
    # text_dir = "output-text"
    # wav_dir = "output-wav"

    text_lst = os.listdir(text_dir)
    wav_lst = os.listdir(wav_dir)

    count_txt = len(text_lst)
    count_wav = len(wav_lst)

    c = 0
    dst_txt_lst = []
    dst_wav_lst = []
    if count_txt == count_wav:
        for i in tqdm(range(count_wav)):
            wav_file = wav_lst[i]
            if args.rename:
                temp_wav_file = "".join(wav_file.split(".")[:-1])
            else:
                temp_wav_file = wav_file
            txt_file = f"{temp_wav_file}.txt"
            if wav_file is None:
                logger.warning(f"{wav_file} not exist.")
                continue
            if txt_file is None:
                logger.warning(f"{txt_file} not exist.")
                continue
            src_txt = f"{text_dir}/{txt_file}"
            src_wav = f"{wav_dir}/{wav_file}"

            prefix_ = str(calc_checksum(wav_file))
            # prefix_ = f"fwav_{(i + 1):06}"
            dst_txt = f"{text_dir}/{prefix_}.txt"
            dst_wav = f"{wav_dir}/{prefix_}.wav"
            if not os.path.exists(src_txt):
                print("Not exists: ", txt_file)
                continue
            if os.path.exists(dst_txt) and os.path.exists(dst_wav):
                print(f"Exist {dst_txt}, {dst_wav}")
                continue

            os.renames(src_txt, dst_txt)
            os.renames(src_wav, dst_wav)

            if not os.path.exists(dst_txt):
                print("Not open file: ", dst_txt)
            with open(dst_txt, 'r') as f:
                lines = " ".join(f.readlines())
                dst_txt_lst.append(lines)
                dst_wav_lst.append(f"{dst_wav}")
                c += 1
    else:
        print("Failed")
    print("Total: ", c)
    assert len(dst_txt_lst) == len(dst_wav_lst)
    sub_df = pd.DataFrame(data={'path': dst_wav_lst, 'transcript': dst_txt_lst})
    sub_df.to_csv(args.data_csv, index=False)
    return sub_df


def split_batch(dataset_dir, sub_df, n_batch, prefix_batch):
    sub_df = sub_df.sample(frac=1).reset_index(drop=True)
    total_samples = len(sub_df)
    print("total number of samples: ", total_samples)
    df_list = np.array_split(sub_df, n_batch)
    for ix, df in enumerate(df_list):
        sub_samples = len(df)
        df.to_csv(f"{dataset_dir}/{prefix_batch}{str(ix + 1)}.csv", index=False)
        logger.info(f"df-{str(ix + 1)}, number of sub-samples: {sub_samples}")


def split_dataset(annotation_df, n_split, train_ratio):
    msk = np.random.rand(len(annotation_df)) <= train_ratio
    # get train csv
    train_csv = annotation_df[msk]
    # get temp test dataframe to split it to test and val
    if n_split == 3:
        temp_test_csv = annotation_df[~msk]
        # get mask train to split temp train dataframe to train and val
        msk_test = np.random.rand(len(temp_test_csv)) <= (1 - train_ratio) / 2
        # split temp train dataframe to train and val
        test_csv = temp_test_csv[msk_test]
        eval_csv = temp_test_csv[~msk_test]
        return train_csv, eval_csv, test_csv
    else:
        test_csv = annotation_df[~msk]
        return train_csv, test_csv


def get_duration_audio(wav_path):
    speech_array, sampling_rate = torchaudio.load(wav_path)
    duration = len(speech_array.squeeze()) / sampling_rate
    return duration


def main():
    args = parse_args()
    if args.rename:
        data_df = rename_files_and_get_annotations(args)
    else:
        data_df = args.data_csv
    output_df = None
    if args.refine_data:
        print("Size original: ", len(data_df))
        audio_path_series = data_df['path']
        data_df['duration'] = audio_path_series.apply(get_duration_audio)

        max_duration = args.max_duration
        min_duration = args.min_duration
        assert max_duration > min_duration

        output_df = data_df[data_df['duration'].apply(lambda x: max_duration >= x >= min_duration)]
        print("Size after refining: ", len(output_df))
        output_df.to_csv(args.refined_data_csv, index=False)

    if output_df is None:
        output_df = data_df

    if args.n_split == 3:
        train_csv, eval_csv, test_csv = split_dataset(output_df, args.n_split, args.train_ratio)
        eval_csv.to_csv(f'{args.dataset_dir}/eval.csv', index=False)
        test_csv.to_csv(f'{args.dataset_dir}/test.csv', index=False)
    else:
        train_csv, test_csv = split_dataset(output_df, args.n_split, args.train_ratio)
        test_csv.to_csv(f'{args.dataset_dir}/test.csv', index=False)
    logger.info(f"split dataset to {args.n_split} set")

    if args.split_batch:
        split_batch(args.dataset_dir, train_csv, args.n_batch, args.prefix_batch)
        logger.info(f"split dataset to {args.n_batch} batch")


if __name__ == "__main__":
    main()
