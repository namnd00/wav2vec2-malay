# coding:utf-8
"""
Name : augment_audio.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/24/2021 8:25 AM
Desc:
"""
import os
import random

import numpy as np
import time
import warnings
from pathlib import Path

from audiomentations import (
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    Resample,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises,
    PolarityInversion,
    Gain,
    Mp3Compression,
    LoudnessNormalization,
    Trim,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
    ApplyImpulseResponse,
    Reverse,
)
from audiomentations.core.audio_loading_utils import load_sound_file

warnings.filterwarnings("ignore")


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, samples, sample_rate=16000):
        for transform in self.transforms:
            augmenter = transform["instance"]
            augmented_samples = augmenter(
                samples=samples, sample_rate=sample_rate
            )
            yield augmented_samples


def generate_compose_transform(aug_dir):
    return [
        {
            "instance": AddBackgroundNoise(
                sounds_path=os.path.join(aug_dir, "background_noises"), p=1.0
            ),
            "num_runs": 1,
        },
        {
            "instance": AddGaussianNoise(
                min_amplitude=0.001, max_amplitude=0.015, p=1.0
            ),
            "num_runs": 1,
        },
        {
            "instance": AddGaussianSNR(p=1.0),
            "num_runs": 1,
            "name": "AddGaussianSNRLegacy",
        },
        {
            "instance": AddGaussianSNR(min_snr_in_db=0, max_snr_in_db=35, p=1.0),
            "num_runs": 1,
            "name": "AddGaussianSNRNew",
        },
        {
            "instance": ApplyImpulseResponse(
                p=1.0, ir_path=os.path.join(aug_dir, "ir")
            ),
            "num_runs": 1,
        },
        {
            "instance": ApplyImpulseResponse(
                p=1.0, ir_path=os.path.join(aug_dir, "ir"), leave_length_unchanged=True
            ),
            "num_runs": 1,
            "name": "AddImpulseResponseLeaveLengthUnchanged",
        },
        {
            "instance": AddShortNoises(
                sounds_path=os.path.join(aug_dir, "short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
                min_time_between_sounds=2.0,
                max_time_between_sounds=4.0,
                burst_probability=0.4,
                min_pause_factor_during_burst=0.01,
                max_pause_factor_during_burst=0.95,
                min_fade_in_time=0.005,
                max_fade_in_time=0.08,
                min_fade_out_time=0.01,
                max_fade_out_time=0.1,
                p=1.0,
            ),
            "num_runs": 1,
        },
        {"instance": BandPassFilter(p=1.0), "num_runs": 1},
        {"instance": ClippingDistortion(p=1.0), "num_runs": 1},
        {
            "instance": FrequencyMask(
                min_frequency_band=0.5, max_frequency_band=0.6, p=1.0
            ),
            "num_runs": 1,
        },
        {"instance": Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0), "num_runs": 1},
        {"instance": HighPassFilter(p=1.0), "num_runs": 1},
        {"instance": LowPassFilter(p=1.0), "num_runs": 1},
        {
            "instance": PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            "num_runs": 1,
        },
        {"instance": LoudnessNormalization(p=1.0), "num_runs": 1},
        {
            "instance": Mp3Compression(backend="lameenc", p=1.0),
            "num_runs": 1,
            "name": "Mp3CompressionLameenc",
        },
        {
            "instance": Mp3Compression(backend="pydub", p=1.0),
            "num_runs": 1,
            "name": "Mp3CompressionPydub",
        },
        {"instance": Normalize(p=1.0), "num_runs": 1},
        {"instance": PolarityInversion(p=1.0), "num_runs": 1},
        {"instance": Resample(p=1.0), "num_runs": 1},
        {"instance": Reverse(p=1.0), "num_runs": 1},
        {
            "instance": Shift(min_fraction=-0.5, max_fraction=0.5, fade=False, p=1.0),
            "num_runs": 1,
            "name": "ShiftWithoutFade",
        },
        {
            "instance": Shift(min_fraction=-0.5, max_fraction=0.5, fade=True, p=1.0),
            "num_runs": 1,
            "name": "ShiftWithShortFade",
        },
        {
            "instance": Shift(
                min_fraction=-0.5,
                max_fraction=0.5,
                rollover=False,
                fade=True,
                fade_duration=0.3,
                p=1.0,
            ),
            "num_runs": 1,
            "name": "ShiftWithoutRolloverWithLongFade",
        },
        # {"instance": TanhDistortion(p=1.0), "num_runs": 5},  # TODO: Uncomment this later
        {"instance": TimeMask(p=1.0), "num_runs": 1},
        {"instance": TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0), "num_runs": 1},
        {"instance": Trim(p=1.0), "num_runs": 1},
    ]


def demo():
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder. Also crudely measure and print execution time.
    """
    DEMO_DIR = ""
    AUG_DIR = ""
    WAV_DIR = ""
    output_dir = os.path.join(DEMO_DIR, "wav_augmentation")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    sound_file_paths = [
        Path(os.path.join(DEMO_DIR, "waves/fp6ABAjcXPI.0007.wav")),
        Path(os.path.join(DEMO_DIR, "waves/fp6ABAjcXPI.0009.wav")),
    ]

    transforms = generate_compose_transform(aug_dir=AUG_DIR)

    for sound_file_path in sound_file_paths:
        samples, sample_rate = load_sound_file(
            sound_file_path, sample_rate=None, mono=False
        )
        if len(samples.shape) == 2 and samples.shape[0] > samples.shape[1]:
            samples = samples.transpose()

        print(
            "Transforming {} with shape {}".format(
                sound_file_path.name, str(samples.shape)
            )
        )
        compose_transform = ComposeTransform(transforms)
        transformed_audio = compose_transform(samples)
        transformed_audio_lst = list(transformed_audio)
        print(len(transformed_audio_lst))
