# coding:utf-8
"""
Name : utils.py
Author : Nam Nguyen
Contact : nam.nd.d3@gmail.com
Time    : 9/26/2021 5:08 PM
Desc:
"""
import logging
import time
from inspect import getframeinfo, stack

import wandb

logger = logging.getLogger(__name__)


class Timer:
    """
    Measure time elapsed since start and since last measurement.
    """

    def __init__(self):
        self.start = self.last = time.perf_counter()

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
        logger.info(
            f"*** TIMER *** - {msg} - Timestamp {timestamp:0.4f} seconds - Duration {duration:0.4f} seconds"
        )
