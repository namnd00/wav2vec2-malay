# Project speech recognition for malay
## Clone git and download datasets

* Git clone

  `git clone https://github.com/nam-nd-d3/wav2vec2-malay`
  
* Install requirements

  `pip install -r requirements.txt`

* Download datasets
  - First:
  `cd datasets/`
  - Next:
  1. https://f000.backblazeb2.com/file/malay-dataset/speech/semisupervised-malay.tar.gz
  
    `16k sample rate, at least 90% voice activity, 93 hours.`
    
  2. https://f000.backblazeb2.com/file/malay-dataset/speech/semisupervised-malay-part2.tar.gz
    
    `16k sample rate, at least 90% voice activity, 70 hours.`
    
  3. https://f000.backblazeb2.com/file/malay-dataset/speech/semisupervised-malay-part3.tar.gz
    
    `16k sample rate, at least 90% voice activity, 59 hours.`
  
## Prepare datasets
  
* Create tokenizer

  `python create_tokenizer.py --data_csv datasets/annotations.csv --path_json_output datasets/vocab.json`
  
* Split dataset to train/val/test

  `python split_data --dataset_dir datasets --data_csv datasets/annotations.csv --split_batch True --n_batch 3 --n_split 3 --train_ratio 0.9 --wav_dir datasets/waves --text_dir datasets/texts`
  
* make sure you're logged into W&B

  `wandb login`
  
* create a sweep -> this will return a sweep id

  `wandb sweep sweep.yaml`

* launch an agent against the sweep

  `wandb agent --count count_number my_sweep_id`
