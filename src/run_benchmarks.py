# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Master code for executing full pipeline with benchmarks
'''

if __name__ == "__main__":
    import argparse
    import logging
    import pathlib
    import warnings
    from gen_synth_data import generate_data
    from data_processing import butterworth_benchmarks
    from train_and_predict import train_model

    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore")

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies")
    parser.add_argument('-s',
                        '--streaming',
                        default=False,
                        action="store_true",
                        help="run streaming inference if true")
    parser.add_argument('-n',
                        '--dataset_len',
                        default=9600,
                        help="number of signals to generate, ideally a multiple of 3")
    parser.add_argument('--save_data',
                        default="True",
                        help="save synthetic data to /data directory if true")
    FLAGS = parser.parse_args()
    dataset_size = int(FLAGS.dataset_len)
    INTEL_FLAG = FLAGS.intel
    STREAMING_FLAG = FLAGS.streaming
    SAVE_FLAG = str(FLAGS.save_data)

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logging.getLogger('matplotlib.*').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    logger.info("Beginning data generation")
    signal_df, train_df, time_dict = generate_data(dataset_size, SAVE_FLAG)
    logger.info("Data generated")

    logger.info("Beginning data processing")
    signals = signal_df.values.reshape((dataset_size//3, 3, 10000))
    feats, np_time, sp_time = butterworth_benchmarks(signals, train_df, dataset_size//3, 0.05)
    logger.info("Data processed")
    logger.info("Total numpy time in data processing: %s", str(np_time))
    logger.info("Total scipy time in data processing: %s", str(sp_time))

    logger.info("Beginning model training and inference")
    rfc, preds, times = train_model(feats, INTEL_FLAG, STREAMING_FLAG)
    preds.to_csv('predictions.csv')
    logger.info("Training and inferencing complete")
    logger.info("Dataset splitting time: %s", str(times["Split"]))
    logger.info("Training time: %s", str(times["Train"]))
    if STREAMING_FLAG:
        logger.info("Real-time inference time: %s", str(times["Inference"]))
    else:
        logger.info("Batch inference time: %s", str(times["Inference"]))
        logger.info("Batch inference accuracy: %s", str(times["Accuracy"]))
        logger.info("Batch inference macro F1: %s", str(times["F1"]))
