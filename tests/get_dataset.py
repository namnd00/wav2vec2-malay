# coding:utf-8
"""
Name : get_dataset.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/10/2021 5:57 PM
Desc:
"""

import shutil
import pandas as pd

path_2 = "../datasets/submission.csv"
sub_df = pd.read_csv(path_2)
test = sub_df[:100]
for d in test['path']:
    print(d)
    shutil.copy(f"../../fine-tuning-wav2vec2/data/{d}", f"../tests/{d}")

test.to_csv('submission.csv', index=False)
