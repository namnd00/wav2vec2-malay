# coding:utf-8
"""
Name : concat_csv.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/10/2021 5:57 PM
Desc:
"""

import pandas as pd

root_path = "submission_1.csv"
addition_path = "submission_2.csv"
root_df = pd.read_csv(root_path)
addition_df = pd.read_csv(addition_path)

df = pd.concat([root_df, addition_df])
print(len(root_df), len(addition_df), len(df))
df.to_csv("submission.csv", index=False)

