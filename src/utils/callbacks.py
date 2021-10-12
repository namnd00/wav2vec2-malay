# coding:utf-8
"""
Name : callbacks.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/26/2021 5:06 PM
Desc:
"""
import logging

from transformers import TrainerCallback
from timer import Timer

logger = logging.getLogger(__name__)
log_timestamp = Timer()


# class LossNaNStoppingCallback(TrainerCallback):
#     """
#     Stops training when loss is NaN.
#
#     Loss is accessed through last logged values so it is useful to set
#     :class:`~transformers.TrainingArguments` argument `logging_steps` to 1.
#     """
#
#     def __init__(self):
#         self.stopped = False
#
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if math.isnan(logs.get("loss", 0.0)):
#             self.stopped = True
#             control.should_training_stop = True
#             logger.info("Loss NaN detected, terminating training")


class TimingCallback(TrainerCallback):
    """
    Logs some interesting timestamps.
    """

    def __init__(self):
        self.training_started = False
        self.evaluation_started = False

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.training_started:
            log_timestamp("trainer ready to start 1st step")

    def on_step_end(self, args, state, control, **kwargs):
        if not self.training_started:
            self.training_started = True
            log_timestamp("first training step")
