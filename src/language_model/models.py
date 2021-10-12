# coding:utf-8
"""
Name    : models.py
Author  : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 12/10/2021
Desc:
"""
import os
import yaml
import logging

from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctcdecode import CTCBeamDecoder

logger = logging.getLogger(__name__)


def create(model_path, lm_path, bin_lm_path):
    if not Path(model_path).is_dir():
        raise Exception(f"{model_path} is not a directory")
    logger.info("loading processor and model wav2vec2")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)

    logger.info("loading config ctc")
    with open(os.path.join(lm_path, "config_ctc.yaml"), 'r') as config_file:
        ctc_lm_params = yaml.load(config_file, Loader=yaml.FullLoader)

    vocab = processor.tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '

    logger.info("create ctc decoder")
    ctcdecoder = CTCBeamDecoder(vocab,
                                model_path='',
                                alpha=0,
                                beta=0,
                                cutoff_top_n=40,
                                cutoff_prob=1.0,
                                beam_width=100,
                                num_processes=4,
                                blank_id=processor.tokenizer.pad_token_id,
                                log_probs_input=True
                                )

    logger.info("create kenlm ctc decoder")
    kenlm_ctcdecoder = CTCBeamDecoder(vocab,
                                      model_path=bin_lm_path,
                                      alpha=ctc_lm_params['alpha'],
                                      beta=ctc_lm_params['beta'],
                                      cutoff_top_n=40,
                                      cutoff_prob=1.0,
                                      beam_width=100,
                                      num_processes=4,
                                      blank_id=processor.tokenizer.pad_token_id,
                                      log_probs_input=True
                                      )

    return processor, model, vocab, ctcdecoder, kenlm_ctcdecoder
