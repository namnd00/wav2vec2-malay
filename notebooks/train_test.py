import os
import gc
import argparse
import soundfile as sf
import pandas as pd
from scipy.io import wavfile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, load_metric
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, processor, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio.
            processor : contain FeatureExtractor and Tokenizer.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df_annotation = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.df_annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir,
                                  self.df_annotation.iloc[idx, 1])
        audio, sr = sf.read(audio_name, dtype='float32')
        if self.transform:
            audio = self.transform(audio)
        transcript = self.df_annotation.iloc[idx, 2].lower()
        nor_audio = self.processor(audio, sampling_rate=sr).input_values
        with self.processor.as_target_processor():
            labels = self.processor([transcript]).input_ids
        sample = {'input_values': nor_audio[0], 'labels': labels[0]}
        return sample


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        #print(features)
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":

    ROOT_DIR = "/content/drive/MyDrive/semisupervised-malay"
    VOCAB_DIR = '/content/drive/MyDrive/vocab.json'

    parser = argparse.ArgumentParser(description= "Traing wav2vec2 model")

    parser.add_argument('--train_data_csv', type=str, required=True, help="str - csv directory path : type (path_file, transcripts)")

    parser.add_argument('--val_data_csv', type=str, required=True, help="str - csv directory path : type (path_file, transcripts)")

    parser.add_argument('--pt_model', type=str, default="facebook/wav2vec2-large-xlsr-53", required=False, help="str - name model pretrain")

    args = parser.parse_args()

    tokenizer = Wav2Vec2CTCTokenizer(args.path_vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    train_data = AudioDataset(csv_file=args.train_data_csv, root_dir=ROOT_DIR, processor=processor)
    val_data = AudioDataset(csv_file=args.val_data_csv, root_dir=ROOT_DIR, processor=processor)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")

    print("-----loading model-----")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.pt_model,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir="My_model",
        group_by_length=True,
        per_device_train_batch_size=4,
        evaluation_strategy="steps",
        num_train_epochs=args.num_epochs,
        fp16=True,
        save_steps=500,
        eval_steps=2000,
        logging_steps=500,
        learning_rate=5e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=1,
    )
    print("---Load pretrain model complete---")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=processor.feature_extractor,)

    print("-----------Training-----------")
    trainer.train()
    trainer.save_model("My_model")



