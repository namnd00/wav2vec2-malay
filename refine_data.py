# coding:utf-8
"""
Name : refine_data.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/28/2021 8:05 PM
Desc:
"""
import argparse
import os

import pandas as pd
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(description='Refine dataset by duration.')
    parser.add_argument('--min_duration',
                        type=int,
                        required=True,
                        help="int - min duration")
    parser.add_argument('--max_duration',
                        type=int,
                        required=True,
                        help="int - max duration")
    parser.add_argument('--input_csv',
                        type=str,
                        required=True,
                        help="csv - path to input dataset")
    parser.add_argument('--output_csv',
                        type=str,
                        required=True,
                        help="csv - path to save refined dataset by duration")

    return parser.parse_args()


def get_duration_audio(wav_path):
    speech_array, sampling_rate = torchaudio.load(wav_path)
    duration = len(speech_array.squeeze()) / sampling_rate
    return duration


def main():
    args = parse_args()
    input_df = pd.read_csv(args.input_csv)
    print("Size original: ", len(input_df))
    audio_path_series = input_df['path']
    input_df['duration'] = audio_path_series.apply(get_duration_audio)

    max_duration = args.max_duration
    min_duration = args.min_duration
    assert max_duration > min_duration

    output_df = input_df[input_df['duration'].apply(lambda x: max_duration >= x >= min_duration)]
    print("Size after refining: ", len(output_df))
    output_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
