# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Master code for executing full pipeline with benchmarks
'''
import os
import argparse
import logging
import pathlib
import warnings
from gen_synth_data import generate_data
from data_processing import butterworth_benchmarks
from train_and_predict import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    warnings.filterwarnings("ignore")

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default=None,
                        help="log file to output benchmarking results to")
    parser.add_argument('-s',
                        '--streaming',
                        action="store_true",
                        help="run streaming inference")
    parser.add_argument('-n',
                        '--dataset_len',
                        type=int,
                        default=9600,
                        help="number of signals to generate, ideally a multiple of 3")
    parser.add_argument('--data_dir',
                        type=str,
                        default=None,
                        help="save synthetic data generated to")

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        help="save outputs to")
    
    FLAGS = parser.parse_args()
    
    if FLAGS.logfile is None:
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG, filemode='w')
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.*').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    logger.info("Beginning data generation")
    signal_df, train_df, time_dict = generate_data(FLAGS.dataset_len, FLAGS.data_dir)
    logger.info("Data generated")

    logger.info("Beginning data processing")
    signals = signal_df.values.reshape((FLAGS.dataset_len//3, 3, 10000))
    feats, np_time, sp_time = butterworth_benchmarks(signals, train_df, FLAGS.dataset_len//3, 0.05)
    logger.info("Data processed")
    logger.info("Total numpy time in data processing: %s", str(np_time))
    logger.info("Total scipy time in data processing: %s", str(sp_time))

    logger.info("Beginning model training and inference")
    rfc, preds, times = train_model(feats, FLAGS.streaming)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    preds.to_csv(os.path.join(FLAGS.output_dir, 'predictions.csv'), header=False)
    logger.info("Training and inferencing complete")
    logger.info("Dataset splitting time: %s", str(times["Split"]))
    logger.info("Training time: %s", str(times["Train"]))
    if FLAGS.streaming:
        logger.info("Real-time inference time: %s", str(times["Inference"]))
    else:
        logger.info("Batch inference time: %s", str(times["Inference"]))
        logger.info("Batch inference accuracy: %s", str(times["Accuracy"]))
        logger.info("Batch inference macro F1: %s", str(times["F1"]))
