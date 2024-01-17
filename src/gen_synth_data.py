# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions for generating synthetic data
'''

import time
import os
import random
import numpy as np
import pandas as pd
from scipy import signal
import timesynth as ts  # pylint: disable=E0401

def neg_sin_0(x):
    return np.sin(x) + (np.sin(4*x))/5 + (np.sin(2*x))/5 + (np.sin(5*x))/4

def neg_sin_1(x):
    return np.sin(x - 2*np.pi/3) + (np.sin(4*(x - 2*np.pi/3)))/4 + (np.sin(2*(x - 2*np.pi/3)))/4 + (np.sin(5*(x - 2*np.pi/3)))/4

def neg_sin_2(x):
    return np.sin(x + 2*np.pi/3) + (np.sin(4*(x + 2*np.pi/3)))/4 + (np.sin(2*(x + 2*np.pi/3)))/4 + (np.sin(5*(x + 2*np.pi/3)))/4

def pos_sin_0(x):
    return np.sin(x) + np.sin(2*np.pi*x)/8

def pos_sin_1(x):
    return np.sin(x + 2*np.pi/3) + np.sin(2*np.pi*(x + 2*np.pi/3))/8

def pos_sin_2(x):
    return np.sin(x - 2*np.pi/3) + np.sin(2*np.pi*(x - 2*np.pi/3))/8

def create_negative_signal(phase):
    global np_time  # pylint: disable=W0601
    start = time.time()
    if phase == 0:
        func = np.frompyfunc(neg_sin_0, 1, 1)
    elif phase == 1:
        func = np.frompyfunc(neg_sin_1, 1, 1)
    else:
        func = np.frompyfunc(neg_sin_2, 1, 1)
    np_time += time.time() - start  # pylint: disable=E0602
    neg_sinusoid = ts.signals.Sinusoidal(amplitude=20, frequency=50, ftype=func)
    neg_white_noise = ts.noise.GaussianNoise(std=2)
    neg_timeseries = ts.TimeSeries(neg_sinusoid, noise_generator=neg_white_noise)
    samples = neg_timeseries.sample(regular_time_samples)[0]
    imp, sptime = neg_target_impulse()
    sig = samples + imp

    return sig, sptime

def create_positive_signal(phase):
    global np_time  # pylint: disable=W0601
    start = time.time()
    if phase == 0:
        func = np.frompyfunc(pos_sin_0, 1, 1)
    elif phase == 1:
        func = np.frompyfunc(pos_sin_1, 1, 1)
    else:
        func = np.frompyfunc(pos_sin_2, 1, 1)
    np_time += time.time() - start
    pos_sinusoid = ts.signals.Sinusoidal(amplitude=15, frequency=50, ftype=func)
    pos_white_noise = ts.noise.GaussianNoise(std=2)
    pos_timeseries = ts.TimeSeries(pos_sinusoid, noise_generator=pos_white_noise)
    samples = pos_timeseries.sample(regular_time_samples)[0]
    imp, sptime = pos_target_impulse()
    sig = samples + imp

    return sig, sptime

def neg_target_impulse():
    start = time.time()
    p_small_impulses = signal.unit_impulse(10000, random.sample(range(10000), 10))*random.sample(range(0, 5), 1)
    n_small_impulses = signal.unit_impulse(10000, random.sample(range(10000), 10))*random.sample(range(-5, 0), 1)
    p_med_impulses = signal.unit_impulse(10000, random.sample(range(10000), 20))*random.sample(range(5, 10), 1)
    n_med_impulses = signal.unit_impulse(10000, random.sample(range(10000), 20))*random.sample(range(-10, -5), 1)
    p_big_impulses = signal.unit_impulse(10000, random.sample(range(10000), 10))*random.sample(range(10, 20), 1)
    n_big_impulses = signal.unit_impulse(10000, random.sample(range(10000), 10))*random.sample(range(-20, -10), 1)
    end = time.time()

    return p_small_impulses + n_small_impulses + p_med_impulses + n_med_impulses + p_big_impulses + n_big_impulses, end-start

def pos_target_impulse():
    start = time.time()
    p_small_impulses = signal.unit_impulse(10000, random.sample(range(10000), 30))*random.sample(range(0, 5), 1)
    n_small_impulses = signal.unit_impulse(10000, random.sample(range(10000), 30))*random.sample(range(-5, 0), 1)
    p_med_impulses = signal.unit_impulse(10000, random.sample(range(10000), 20))*random.sample(range(5, 15), 1)
    n_med_impulses = signal.unit_impulse(10000, random.sample(range(10000), 20))*random.sample(range(-15, -5), 1)
    p_big_impulses = signal.unit_impulse(10000, random.sample(range(10000), 10))*random.sample(range(15, 25), 1)
    n_big_impulses = signal.unit_impulse(10000, random.sample(range(10000), 10))*random.sample(range(-25, -15), 1)
    p_max_impulses = signal.unit_impulse(10000, random.sample(range(10000), 5))*random.sample(range(25, 50), 1)
    n_max_impulses = signal.unit_impulse(10000, random.sample(range(10000), 5))*random.sample(range(-50, -25), 1)
    end = time.time()

    return p_small_impulses + n_small_impulses + p_med_impulses + n_med_impulses + p_big_impulses + n_big_impulses + \
        p_max_impulses + n_max_impulses, end-start


def generate_data(dataset_size, data_dir):
    global np_time, regular_time_samples  # pylint: disable=W0601
    time_sampler = ts.TimeSampler(stop_time=0.02)
    regular_time_samples = time_sampler.sample_regular_time(num_points=10000)

    sig_dict = {}
    target_id = []
    group_id = []
    phase_id = []

    num_groups = dataset_size//3
    negative_count = (num_groups//16)*15
    positive_count = num_groups - negative_count
    count = 0

    total_sp = 0
    np_time = 0

    for i in range(negative_count):
        for phase in range(3):
            group_id.append(count//3)
            phase_id.append(phase)
            target_id.append(0)
            sig_dict[str(count)], spt = create_negative_signal(phase)
            total_sp += spt
            count += 1

    for i in range(positive_count):
        for phase in range(3):
            group_id.append(count//3)
            phase_id.append(phase)
            target_id.append(1)
            sig_dict[str(count)], spt = create_positive_signal(phase)
            total_sp += spt
            count += 1

    metadata_df = pd.DataFrame()
    metadata_df["Sig_id"] = [i for i in range(count)]  # pylint: disable=R1721
    metadata_df["Group_id"] = group_id
    metadata_df["Phase"] = phase_id
    metadata_df["Target"] = target_id

    signal_df = pd.DataFrame(sig_dict)

    time_dict = {"SP": total_sp, "NP": np_time}

    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)
        signal_df.to_csv(os.path.join(data_dir, 'signal_data.csv'), index=False)
        metadata_df.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)
    return signal_df, metadata_df, time_dict
