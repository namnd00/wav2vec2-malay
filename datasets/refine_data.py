import argparse
import os
import pandas as pd
import hashlib
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Refine dataset.')
    parser.add_argument('--data_csv',
                        type=str,
                        required=True,
                        help="str - csv directory path")
    parser.add_argument('--n_batch',
                        type=int,
                        required=True,
                        help="int - number of batch")
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
                        default='batch_',
                        required=False,
                        help="str - prefix of file batch, i.e: batch_0.csv")

    return parser.parse_args()


def calc_checksum(f):
    return str(hashlib.sha3_512(f.encode('utf-8')).hexdigest()[0:32])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    args = parse_args()
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
    print("Total: ", c)
    assert len(dst_txt_lst) == len(dst_wav_lst)
    sub_df = pd.DataFrame(data={'path': dst_wav_lst, 'transcript': dst_txt_lst})
    sub_df.to_csv(args.data_csv, index=False)

    sub_df = sub_df.sample(frac=1).reset_index(drop=True)
    total_samples = len(sub_df)
    logger.info("total number of samples: ", total_samples)
    df_list = np.vsplit(sub_df, args.n_batch)
    for ix, df in enumerate(df_list):
        sub_samples = len(df)
        df.to_csv(f"{args.prefix_batch}{str(ix+1)}.csv", index=False)
        logger.info(f"df-{str(ix+1)}, number of sub-samples: {sub_samples}")


if __name__ == "__main__":
    main()


