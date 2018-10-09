#!/usr/bin/python
# -*- coding: latin-1 -*-

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# * * * * * * * * * * * * * *  Taken From * * * * * * * * * * * * * * * * * * #
# * https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/loss.py * #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import numpy as np

def create_linear_decay_fn(initial_value, final_value, max_step):
    def decay_fn(step):
        relative = 1. - step / max_step
        return initial_value * relative + final_value * (1. - relative)

    return decay_fn


def create_cycle_decay_fn(initial_value, final_value, cycle_len, num_cycles):
    max_step = cycle_len * num_cycles

    def decay_fn(step):
        relative = 1. - step / max_step
        relative_cosine = 0.5 * (np.cos(np.pi * np.mod(step, cycle_len) / cycle_len) + 1.0)
        return relative_cosine * (initial_value - final_value) * relative + final_value

    return decay_fn


def create_decay_fn(decay_type, **kwargs):
    if decay_type == "linear":
        return create_linear_decay_fn(**kwargs)
    elif decay_type == "cycle":
        return create_cycle_decay_fn(**kwargs)
    else:
        raise NotImplementedError()
