import os
import pandas as pd
import shutil
import hashlib
from tqdm import tqdm
from pathlib import Path
import random
import logging

logger = logging.getLogger(__name__)

text_dir = "datasets/output-text"
wav_dir = "datasets/waves"

text_lst = os.listdir(text_dir)
wav_lst = os.listdir(wav_dir)

count_txt = len(text_lst)
count_wav = len(wav_lst)

c = 0
dst_txt_lst = []
dst_wav_lst = []


def calc_checksum(f):
    return str(hashlib.sha3_512(f.encode('utf-8')).hexdigest()[0:32])


if count_txt == count_wav:
    for i in tqdm(range(count_wav)):
        wav_file = wav_lst[i]
        txt_file = f"{wav_file}.txt"
        if wav_file is None:
            logger.warning(f"{wav_file} not exist.")
            continue
        if txt_file is None:
            logger.warning(f"{txt_file} not exist.")
            continue
        src_txt = f"{text_dir}/{txt_file}"
        src_wav = f"{wav_dir}/{wav_file}"

        prefix_ = str(calc_checksum(wav_file))
        dst_txt = f"{text_dir}/{prefix_}.txt"
        dst_wav = f"{wav_dir}/{prefix_}.wav"

        if not os.path.exists(src_txt):
            logger.info("Not exists: ", txt_file)
            continue
        if os.path.exists(dst_txt) and os.path.exists(dst_wav):
            logger.info(f"Exist {dst_txt}, {dst_wav}")
            continue
        os.rename(src_txt, dst_txt)
        os.rename(src_wav, dst_wav)

        if not os.path.exists(dst_txt):
            logging.error("Not open file: ", dst_txt)
        f = open(dst_txt, 'r')
        try:
            lines = " ".join(f.readlines())
            dst_txt_lst.append(lines)
            dst_wav_lst.append(f"datasets/{dst_wav}")
            c += 1
        finally:
            f.close()
else:
    print("Failed")
print(c)
if len(dst_txt_lst) == len(dst_wav_lst):
    sub_df = pd.DataFrame(data={'path': dst_wav_lst, 'transcript': dst_txt_lst})
    sub_df.to_csv('waves.csv', index=False)
else:
    print("Failed")
