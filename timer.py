# coding:utf-8
"""
Name : timer.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/10/2021 10:07 AM
Desc:
"""
import time
import logging
import wandb

from inspect import getframeinfo, stack


class Timer:
    """
    Measure time elapsed since start and since last measurement.
    """

    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.logger = logging.getLogger(__name__)

    def __call__(self, msg=None):
        t = time.perf_counter()
        timestamp = t - self.start
        duration = t - self.last
        self.last = t
        if msg is None:
            caller = getframeinfo(stack()[1][0])
            msg = f"line {caller.lineno}"
        wandb.log({f"timestamps/{msg}": timestamp})
        wandb.log({f"durations/{msg}": duration})
        self.logger.info(
            f"*** TIMER *** - {msg} - Timestamp {timestamp:0.4f} seconds - Duration {duration:0.4f} seconds"
        )
