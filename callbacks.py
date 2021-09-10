# coding:utf-8
"""
Name : callbacks.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/10/2021 9:23 AM
Desc:
"""
import numpy as np
import logging

from transformers import TrainerCallback
from timer import Timer


class LossNaNStoppingCallback(TrainerCallback):
    """
    Stops training when loss is NaN.

    Loss is accessed through last logged values so it is useful to set
    :class:`~transformers.TrainingArguments` argument `logging_steps` to 1.
    """

    def __init__(self):
        self.stopped = False
        self.logger = logging.getLogger(__name__)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if np.isnan(logs.get("loss", 0.0)):
            self.stopped = True
            control.should_training_stop = True
            self.logger.info("Loss NaN detected, terminating training")


class TimingCallback(TrainerCallback):
    """
    Logs some interesting timestamps.
    """

    def __init__(self):
        self.training_started = False
        self.evaluation_started = False
        self.log_timestamp = Timer()

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.training_started:
            self.log_timestamp("trainer ready to start 1st step")

    def on_step_end(self, args, state, control, **kwargs):
        if not self.training_started:
            self.training_started = True
            self.log_timestamp("first training step")
