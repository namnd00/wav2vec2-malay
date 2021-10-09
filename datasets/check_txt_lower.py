import os
from tqdm import tqdm
import re
path = "transcripts.txt"

with open(path, 'r') as file:
	for f in tqdm(file):
		rs = bool(re.search(r'\d', f))
		if rs:
			print(f)
