# coding:utf-8
"""
Name : augment_audio.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/24/2021 8:25 AM
Desc:
"""
import os
import warnings
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
warnings.filterwarnings("ignore")


def get_transforms(aug_dir):
    return [
        AddBackgroundNoise(
            sounds_path=os.path.join(aug_dir, "background_noises"), p=1.0
        ),
        AddGaussianNoise(
            min_amplitude=0.001, max_amplitude=0.015, p=1.0
        ),
        AddGaussianSNR(p=1.0),
        AddGaussianSNR(min_snr_in_db=0, max_snr_in_db=35, p=1.0),
        AddShortNoises(
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
        BandPassFilter(p=1.0),
        ClippingDistortion(p=1.0),
        FrequencyMask(
            min_frequency_band=0.5, max_frequency_band=0.6, p=1.0
        ),
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0),
        HighPassFilter(p=1.0),
        LowPassFilter(p=1.0),
        PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
        LoudnessNormalization(p=1.0),
        Normalize(p=1.0),
        PolarityInversion(p=1.0),
        Resample(p=1.0),
        Reverse(p=1.0),
        Shift(min_fraction=-0.5, max_fraction=0.5, fade=False, p=1.0),
        Shift(min_fraction=-0.5, max_fraction=0.5, fade=True, p=1.0),
        Shift(
            min_fraction=-0.5,
            max_fraction=0.5,
            rollover=False,
            fade=True,
            fade_duration=0.3,
            p=1.0,
        ),
        TimeMask(p=1.0),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        Trim(p=1.0),
    ]
