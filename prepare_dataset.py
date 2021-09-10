# coding:utf-8
"""
Name : prepare_dataset.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/10/2021 9:30 AM
Desc:
"""
import torchaudio
import re
import logging

from unidecode import unidecode
from transforms.spec_augment import TimeMask, FrequencyMask, Compose

logger = logging.getLogger(__name__)


def remove_special_characters(batch, chars_to_ignore_regex, train=True):
    batch["text"] = (
        re.sub(chars_to_ignore_regex, "", unidecode(batch["sentence"]))
            .lower()
            .strip()
    )
    if train:
        batch["text"] += " "
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def get_resampler(sampling_rate, resampler=None):
    if sampling_rate in resampler.keys():
        return resampler[sampling_rate]
    else:
        logger.info(f"Creating new resampler for {sampling_rate}")
        resampler[sampling_rate] = torchaudio.transforms.Resample(
            sampling_rate, 16_000
        )
        return resampler[sampling_rate]


def get_spec_augment(max_width=10, use_mean=False):
    return Compose([
        FrequencyMask(max_width=max_width, use_mean=use_mean),
        TimeMask(max_width=max_width, use_mean=use_mean),
    ])


def get_noise_speech(noise_path=None):
    pass


# Preprocessing the datasets.
# We need to read the audio files as arrays and tokenize the targets.
def speech_file_to_array_fn(batch,
                            spec_augment=True,
                            add_noise=False,
                            transforms=None,
                            max_width=10,
                            use_mean=False):
    if spec_augment:
        transforms = get_spec_augment(max_width, use_mean)
    if add_noise:
        pass
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    if transforms is not None:
        speech_array = transforms(speech_array)
    batch["speech"] = get_resampler(sampling_rate)(speech_array).squeeze().numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["text"]
    batch["duration"] = len(speech_array.squeeze()) / sampling_rate
    return batch


# def augment_add_noise_speech_file_to_array_fn(batch, augment):
#     speech_array, sampling_rate = torchaudio.load(batch["path"])
#     len_augment = len(augment)
#     id_rand = np.random.randint(int(len_augment * 0.9))
#     speech_array = speech_array + augment[id_rand: len(speech_array) + id_rand] * 0.3
#     batch["speech"] = speech_array
#     batch["sampling_rate"] = sampling_rate
#     batch["transcript"] = batch["transcript"].lower()
#     batch["target_text"] = batch["transcript"]
#     return batch


def filter_by_duration(batch):
    return (
            10 >= batch["duration"] >= 1
            and len(batch["target_text"]) > 5
    )  # about 98% of samples


def prepare_dataset(batch, processor):
    # check that all files have the correct sampling rate
    assert (
            len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values
    # Setup the processor for targets
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


def get_length(item):
    # speeds up grouping by length in pre-loaded dataset
    item["length"] = len(item["input_values"])
    return item
