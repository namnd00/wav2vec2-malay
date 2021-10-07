import soundfile as sf
import numpy as np
import argparse
import pandas as pd
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, load_metric

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer


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
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
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


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read("datasets/waves/" + batch["path"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["transcript"] = batch["transcript"].lower()
    batch["target_text"] = batch["transcript"]
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


def prepare_dataset(batch):
    assert (
            len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model wav2vec2.')
    parser.add_argument('--path_csv_train',
                        type=str,
                        required=False,
                        help="str - csv directory path : type (path_file, transcripts)")

    parser.add_argument('--path_csv_test',
                        type=str,
                        required=False,
                        help="str - csv directory path : type (path_file, transcripts)")

    parser.add_argument('--path_vocab',
                        type=str,
                        default='vocab.json',
                        required=False,
                        help="str - json directory path")

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        required=False,
                        help="int - batch size")

    parser.add_argument('--model_pretrain',
                        type=str,
                        default="facebook/wav2vec2-large-xlsr-53",
                        required=False,
                        help="str - name model pretrain")

    parser.add_argument('--model_output',
                        type=str,
                        default="my_model",
                        required=False,
                        help="str - path model output")

    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        required=False,
                        help="int - num train epochs")

    args = parser.parse_args()
    path_csv_train = "datasets/train.csv"
    path_csv_test = "datasets/test.csv"
    path_vocab = "datasets/vocab.json"
    print('load dataset ...')
    data_train = pd.read_csv(path_csv_train, encoding='utf-8')
    data_test = pd.read_csv(path_csv_test, encoding='utf-8')
    train_data = Dataset.from_pandas(data_train)
    test_data = Dataset.from_pandas(data_test)

    tokenizer = Wav2Vec2CTCTokenizer(path_vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=False, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    print('preprocessing data ..')
    train_data = train_data.map(speech_file_to_array_fn, num_proc=1)
    test_data = test_data.map(speech_file_to_array_fn, num_proc=1)

    train_data = train_data.map(prepare_dataset, batch_size=4, num_proc=1, batched=True)
    test_data = test_data.map(prepare_dataset, batch_size=4, num_proc=1, batched=True)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")
    print('load pretrain ..')
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_pretrain,
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
    # model.to('cuda')
    model.freeze_feature_extractor()
    training_args = TrainingArguments(
        output_dir=args.model_output,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        fp16=True,
        save_steps=500,
        eval_steps=2000,
        logging_steps=500,
        learning_rate=5e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(args.model_output)
