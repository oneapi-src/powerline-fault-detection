# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions needed for reading and processing data
'''
import time
import numpy as np
from scipy import signal

def peak_features(sig):
    global np_time, sp_time  # pylint: disable=W0601

    start = time.time()
    p_peaks_all = signal.find_peaks(sig, height=None)[0]
    n_peaks_all = signal.find_peaks(-sig, height=None)[0]
    sp_time += (time.time() - start)  # pylint: disable=E0602

    start = time.time()
    pn_peaks_all = np.union1d(p_peaks_all, n_peaks_all)
    np_time += (time.time() - start)  # pylint: disable=E0602

    maxDistance = 10
    maxHeightRatio = 0.25
    maxTicksRemoval = 500

    start = time.time()
    peak_ind_dist = np.diff(pn_peaks_all)
    np_time += (time.time() - start)

    sym_peak_pair_indices = []
    for ind, dist in enumerate(peak_ind_dist):
        ratio = sig[pn_peaks_all[ind+1]]/sig[pn_peaks_all[ind]]
        if dist < maxDistance:
            if ratio < 0:
                start = time.time()
                ratio_check = maxHeightRatio < np.abs(ratio) < (1/maxHeightRatio)
                np_time += (time.time() - start)
                if ratio_check:
                    sym_peak_pair_indices.append((pn_peaks_all[ind], pn_peaks_all[ind+1]))

    false_peaks = []
    for peak_val in pn_peaks_all:
        for sym_peak_pair_ind_curr, sym_peak_pair_ind_next in sym_peak_pair_indices:  # pylint: disable=W0612
            if sym_peak_pair_ind_next <= peak_val <= (sym_peak_pair_ind_next + maxTicksRemoval):
                false_peaks.append(peak_val)

    start = time.time()
    false_peaks = np.unique(false_peaks)
    true_peaks = np.setdiff1d(pn_peaks_all, false_peaks)
    p_peaks = np.intersect1d(p_peaks_all, true_peaks)
    n_peaks = np.intersect1d(n_peaks_all, true_peaks)
    if true_peaks.size > 0:
        true_peaks_sd = np.std(true_peaks)
        np_time += (time.time() - start)
    else:
        np_time += (time.time() - start)
        true_peaks_sd = None
    
    p_true_peak_heights = []
    for ind in p_peaks:
        p_true_peak_heights.append(sig[ind])
    n_true_peak_heights = []
    for ind in n_peaks:
        n_true_peak_heights.append(sig[ind])

    if len(p_true_peak_heights) > 0:
        start = time.time()
        p_mean_height = np.mean(p_true_peak_heights)
        p_sd_height = np.std(p_true_peak_heights)
        p_min_height = np.min(p_true_peak_heights)
        p_max_height = np.max(p_true_peak_heights)
        np_time += (time.time() - start)
        p_heights_stats = [p_mean_height, p_min_height, p_max_height, p_sd_height]
    else:
        p_heights_stats = [None, None, None, None]

    if len(n_true_peak_heights) > 0:
        start = time.time()
        n_mean_height = np.mean(n_true_peak_heights)
        n_sd_height = np.std(n_true_peak_heights)
        n_min_height = np.min(n_true_peak_heights)
        n_max_height = np.max(n_true_peak_heights)
        np_time += (time.time() - start)
        n_heights_stats = [n_mean_height, n_min_height, n_max_height, n_sd_height]
    else:
        n_heights_stats = [None, None, None, None]

    start = time.time()
    true_widths = signal.peak_widths(sig, true_peaks, rel_height=1)[0]
    sp_time += (time.time()-start)

    if true_widths.size > 0:
        nonzero_widths = true_widths[true_widths != 0]
        start = time.time()
        true_mean_width = np.mean(true_widths)
        true_sd_width = np.std(true_widths)
        
        if nonzero_widths.size > 0:
            true_min_width = np.min(nonzero_widths)
        else:
            true_min_width = np.min(true_widths)
        true_max_width = np.max(true_widths)
        np_time += (time.time() - start)
        width_stats = [true_mean_width, true_min_width, true_max_width, true_sd_width]
    else:
        width_stats = [None, None, None, None]

    return p_peaks.size, n_peaks.size, true_peaks_sd, p_heights_stats, n_heights_stats, width_stats

def unfiltered_features(sig):
    m = np.mean(sig, axis=0)
    sd = np.std(sig, axis=0, ddof=0)

    return np.where(sd == 0, 0, m / sd)


def features_to_df(sig_features, train_df):
    df2 = train_df.assign(Signal_to_Noise_Ratio=sig_features['noise_ratio'])
    df2['Num_of_Positive_True_Peaks'] = sig_features['num_pos_peaks']
    df2['Num_of_Negative_True_Peaks'] = sig_features['num_neg_peaks']
    df2['Std_Dev_of_True_Peak_Positions'] = sig_features['true_peaks_sd']
    df2['Mean_of_Positive_True_Peak_Heights'] = sig_features['p_mean_heights']
    df2['Min_of_Positive_True_Peak_Heights'] = sig_features['p_min_heights']
    df2['Max_of_Positive_True_Peak_Heights'] = sig_features['p_max_heights']
    df2['Std_Dev_of_Positive_True_Peak_Heights'] = sig_features['p_sd_heights']
    df2['Mean_of_Negative_True_Peak_Heights'] = sig_features['n_mean_heights']
    df2['Min_of_Negative_True_Peak_Heights'] = sig_features['n_min_heights']
    df2['Max_of_Negative_True_Peak_Heights'] = sig_features['n_max_heights']
    df2['Std_Dev_of_Negative_True_Peak_Heights'] = sig_features['n_sd_heights']
    df2['Mean_of_True_Peak_Widths'] = sig_features['mean_widths']
    df2['Min_of_True_Peak_Widths'] = sig_features['min_widths']
    df2['Max_of_True_Peak_Widths'] = sig_features['max_widths']
    df2['Std_Dev_of_True_Peak_Widths'] = sig_features['sd_widths']

    return df2


def butterworth_benchmarks(signals, metadata, dataset_size, Wn):
    global np_time, sp_time  # pylint: disable=W0601
    np_time = 0
    sp_time = 0

    sig_features = {'noise_ratio': [], 'num_pos_peaks': [], 'num_neg_peaks': [], 'true_peaks_sd': [], 'p_mean_heights': [], 'p_min_heights': [],
                    'p_max_heights': [], 'p_sd_heights': [], 'n_mean_heights': [], 'n_min_heights': [], 'n_max_heights': [], 'n_sd_heights': [],
                    'mean_widths': [], 'min_widths': [], 'max_widths': [], 'sd_widths': []}

    for group in range(dataset_size):
        for phase in range(3):
            sig = signals[group, phase, :]

            start = time.time()
            noise_feats = unfiltered_features(sig)
            np_time += (time.time() - start)
            sig_features['noise_ratio'].append(noise_feats)

            start = time.time()
            b, a = signal.butter(3, Wn)
            filtered = signal.filtfilt(b, a, sig)
            sp_time += (time.time()-start)

            num_pos, num_neg, peak_sd, p_heights, n_heights, widths = peak_features(filtered)
            sig_features['num_pos_peaks'].append(num_pos)
            sig_features['num_neg_peaks'].append(num_neg)
            sig_features['true_peaks_sd'].append(peak_sd)
            sig_features['p_mean_heights'].append(p_heights[0])
            sig_features['p_min_heights'].append(p_heights[1])
            sig_features['p_max_heights'].append(p_heights[2])
            sig_features['p_sd_heights'].append(p_heights[3])
            sig_features['n_mean_heights'].append(n_heights[0])
            sig_features['n_min_heights'].append(n_heights[1])
            sig_features['n_max_heights'].append(n_heights[2])
            sig_features['n_sd_heights'].append(n_heights[3])
            sig_features['mean_widths'].append(widths[0])
            sig_features['min_widths'].append(widths[1])
            sig_features['max_widths'].append(widths[2])
            sig_features['sd_widths'].append(widths[3])

    return features_to_df(sig_features, metadata), np_time, sp_time
