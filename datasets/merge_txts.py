import os
from tqdm import tqdm
from glob import glob

filenames = glob(f"output-text/*.txt")
print(f"Total {len(filenames)} text files.")
with open('transcripts.txt', 'w') as outfile:
    for fname in tqdm(filenames):
        if not os.path.exists(fname) and os.path.isdir(fname):
            continue
        with open(fname) as infile:
            for line in infile:
                outfile.write(f"{line.lower()}\n")
