import os
from tqdm import tqdm
from glob import glob

path = "texts"
filenames = glob("texts/*.txt")
print(f"Total {len(filenames)} text files.")
with open('transcripts.txt', 'w') as outfile:
    for fname in tqdm(filenames):
        with open(fname) as infile:
            for line in infile:
                outfile.write(f"{line.lower()}\n")
