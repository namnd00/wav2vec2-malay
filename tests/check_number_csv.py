# coding:utf-8
"""
Name : check_number_csv.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/30/2021 9:28 AM
Desc:
"""
import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="count number sample in each csv file")
    parser.add_argument("--dir", help="path to dir contain csv files", required=True, default="datasets")
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.dir
    files_n_folders_lst = os.listdir(path)
    print(f"->Total file and folder in {path}:\n{files_n_folders_lst}")
    csv_files = [f for f in files_n_folders_lst if ".csv" in f]
    print(f"->Existing {len(csv_files)} csv files: {csv_files}")
    for ix, csv_file in enumerate(csv_files):
        print("=="*100)
        csv_path = os.path.join(path, csv_file)
        df = pd.read_csv(csv_path)
        print(f"{ix+1}. {csv_file} -> Columns: {list(df.columns)} -> Samples: {len(df)}")
        print(df.head())


if __name__ == "__main__":
    main()
