# coding:utf-8
"""
Name    : evaluate.py
Author  : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 11/10/2021
Desc:
"""
import os

import pandas as pd
import torch
import torchaudio

from language_model import models
from argparse import ArgumentParser, RawTextHelpFormatter
from datasets import Dataset, load_metric

DESCRIPTION = """
Much of the code in this file was lifted from a HuggingFace blog entry:
Fine-Tune XLSR-Wav2Vec2 for low-resource ASR with Transformers
https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
by Patrick von Platen
An implementation of a CTC (Connectionist Temporal Classification) beam search decoder with
KenLM language models support from https://github.com/parlance/ctcdecode has been added.
"""


# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["transcript"] = batch["transcript"].strip()  # + " "
    batch["speech"] = speech_array
    return batch


def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)

    batch["pred_strings"] = processor.batch_decode(pred_ids)[0].strip()

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)
    pred_with_ctc = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
    batch["pred_strings_with_ctc"] = pred_with_ctc.strip()

    beam_results, beam_scores, timesteps, out_lens = kenlm_ctcdecoder.decode(logits)
    pred_with_lm = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
    batch["pred_strings_with_lm"] = pred_with_lm.strip()

    return batch


def main(wav2vec2_model_path,
         dataset_dir,
         test_dataset_path,
         lm_path,
         bin_lm_path,
         benchmarks,
         batch_size=8,
         **args):
    global processor
    global model
    global vocab
    global ctcdecoder
    global kenlm_ctcdecoder

    processor, model, vocab, ctcdecoder, kenlm_ctcdecoder = models.create(wav2vec2_model_path,
                                                                          lm_path,
                                                                          bin_lm_path)

    test_dataset_df = pd.read_csv(test_dataset_path)
    test_dataset_df['path'] = dataset_dir + "/" + test_dataset_df['path']
    test_dataset = Dataset.from_pandas(test_dataset_df)

    wer = load_metric("wer")

    model.to("cuda")

    test_dataset = test_dataset.map(speech_file_to_array_fn)
    result = test_dataset.map(evaluate, batch_size=batch_size)

    print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["transcript"])))
    print("WER with CTC: {:2f}".format(
        100 * wer.compute(predictions=result["pred_strings_with_ctc"], references=result["transcript"])))
    print("WER with CTC+LM: {:2f}".format(
        100 * wer.compute(predictions=result["pred_strings_with_lm"], references=result["transcript"])))

    result_csv = "wer_ctc_lm.csv"
    print(f"Export to {result_csv}")
    result.to_csv(f"{benchmarks}/{result_csv}", index=False)


if __name__ == "__main__":

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    parser.add_argument("--lm_path", required=True,
                        dest="lm_path", default="pretrained_models/malay_lm")

    parser.add_argument("--bin_lm_path", required=True,
                        dest="bin_lm_path", default="pretrained_models/malay_lm/4gram_malay_lm.binary")

    parser.add_argument("--wav2vec2_model_path", required=True,
                        dest="wav2vec2_model_path", default="pretrained_models/")

    parser.add_argument("--dataset_dir", required=True,
                        dest="dataset_dir", default="datasets/waves")

    parser.add_argument("--test_dataset_path", required=True,
                        dest="test_dataset_path", default="datasets/test.csv")

    parser.add_argument("--batch_size",
                        dest="batch_size", default=8)

    parser.add_argument("--benchmarks",
                        dest="benchmarks", default="benchmarks")

    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))