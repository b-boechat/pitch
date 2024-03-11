import argparse as ap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from definitions import DEFAULT_BATCH, DEFAULT_ONSET_POSITIVE_WEIGHT, \
    DEFAULT_SHUFFLE_BUFFER, DEFAULT_LEARNING_RATE, DEFAULT_LABEL_SMOOTHING, \
    CQT_PROCESSED_BASE_PATH, DEFAULT_EPOCHS, \
    DEFAULT_ONSET_THRESHOLD_LIST, DEFAULT_FRAME_THRESHOLD_LIST, \
    SAVED_MODELS_BASE_PATH
from train import train
from cross_validate import cross_validate
from read_evaluation import read_metrics
from evaluate_model import evaluate_model

def train_wrapper(args):
    if args.verbosity >= 1:
        print(args)

    with tf.device(f'/GPU:{args.gpu}'):
        train(
            learning_rate = args.learning_rate,
            label_smoothing = args.label_smoothing,
            buffer_size = args.shuffle_buffer,
            batch_size = args.batch_size,
            onset_positive_weight = args.onset_positive_weight,
            epochs = args.epochs,
            verbose = args.verbosity,
            data_base_dir = args.data_base_dir,
            output_folder_id = args.output_folder_id,
            save_history = args.save_history,
            output_base_path = SAVED_MODELS_BASE_PATH
        )

def evaluate_model_wrapper(args):
    if args.verbosity >= 1:
        print(args)

    with tf.device(f'/GPU:{args.gpu}'):
        for onset_threshold in args.onset_threshold_list:
            for frame_threshold in args.frame_threshold_list:
                evaluate_model(
                    model_id = args.model_id, 
                    split_name = args.split_name, 
                    onset_threshold = onset_threshold,
                    frame_threshold = frame_threshold,
                    base_path = args.base_path,
                    verbosity = args.verbosity
                )

def read_metrics_wrapper(args): # TODO add other arguments
    read_metrics(args.model_id, split_name=args.split_name)

def cross_validate_wrapper(args):
    if args.verbosity >= 1:
        print(args)

    with tf.device(f'/GPU:{args.gpu}'):
        cross_validate(
            learning_rate = args.learning_rate,
            label_smoothing = args.label_smoothing,
            buffer_size = args.shuffle_buffer,
            batch_size = args.batch_size,
            onset_positive_weight = args.onset_positive_weight,
            epochs = args.epochs,
            verbose = args.verbosity,
            data_base_dir = args.data_base_dir,
            output_cv_folder_id = args.output_cv_folder_id,
            save_history = args.save_history,
            num_cv_groups = args.num_cv_groups,
            output_base_path = SAVED_MODELS_BASE_PATH,
            model_index_to_resume = args.model_index_to_resume
        )

def parse_console():
    parser = ap.ArgumentParser(description="Interface para treinamento e avaliação do sistema de detecção de frequência fundamental baseado no \"basic pitch\"")

    subparsers = parser.add_subparsers()

    # "train" parser
    sp_train = subparsers.add_parser("train", aliases="t")
    sp_train.set_defaults(func=train_wrapper)
    sp_train.add_argument("-r", "--lr", dest="learning_rate", type=float, metavar="LEARNING_RATE", default=DEFAULT_LEARNING_RATE)
    sp_train.add_argument("-s", "--shuffle", dest="shuffle_buffer", type=int, metavar="SHUFFLE_BUFFER_SIZE", default=DEFAULT_SHUFFLE_BUFFER)
    sp_train.add_argument("-b", "--batch", dest="batch_size", type=int, metavar="BATCH_SIZE", default=DEFAULT_BATCH)
    sp_train.add_argument("-e", "--epochs", dest="epochs", type=int, metavar="EPOCHS", default=DEFAULT_EPOCHS)
    sp_train.add_argument("-l", "--smoothing", dest="label_smoothing", type=float, metavar="LABEL_SMOOTHING", default=DEFAULT_LABEL_SMOOTHING)
    sp_train.add_argument("-w", "--weight", dest="onset_positive_weight", type=float, default=DEFAULT_ONSET_POSITIVE_WEIGHT)
    sp_train.add_argument("-i", "--data_base_dir", dest="data_base_dir", default=CQT_PROCESSED_BASE_PATH)
    #sp_train.add_argument("-n", "--no_test", dest="no_test", action="store_true")
    sp_train.add_argument("-v", "--verbosity", dest="verbosity", action="count", default=0)
    sp_train.add_argument("-o", "--output_folder", dest="output_folder_id", default=None)
    sp_train.add_argument("-y", "--dont_save_history", dest="save_history", action="store_false")
    sp_train.add_argument("-g", "--gpu", dest="gpu", default=1)

    # "evaluate_model" parser
    sp_evaluate = subparsers.add_parser("evaluate", aliases="e")
    sp_evaluate.set_defaults(func=evaluate_model_wrapper)
    sp_evaluate.add_argument("model_id")
    sp_evaluate.add_argument("-s", "--split", dest="split_name", default="val")
    sp_evaluate.add_argument("-o", "--onset_threshold_list", dest="onset_threshold_list", type=float, nargs='+', default=DEFAULT_ONSET_THRESHOLD_LIST)
    sp_evaluate.add_argument("-f", "--frame_threshold_list", dest="frame_threshold_list", type=float, nargs='+', default=DEFAULT_FRAME_THRESHOLD_LIST)
    sp_evaluate.add_argument("-b", "--base_path", dest="base_path", default=SAVED_MODELS_BASE_PATH)
    sp_evaluate.add_argument("-v", "--verbosity", dest="verbosity", action="count", default=0)
    sp_evaluate.add_argument("-g", "--gpu", dest="gpu", default=1)

    # "read_eval parser"
    sp_read_eval = subparsers.add_parser("read_eval", aliases="r")
    sp_read_eval.set_defaults(func=read_metrics_wrapper)
    sp_read_eval.add_argument("model_id")
    sp_read_eval.add_argument("-s", "--split", dest="split_name", default="test")

    sp_train_cv = subparsers.add_parser("train_cv", aliases="tc")
    sp_train_cv.set_defaults(func=cross_validate_wrapper)
    sp_train_cv.add_argument("-r", "--lr", dest="learning_rate", type=float, metavar="LEARNING_RATE", default=DEFAULT_LEARNING_RATE)
    sp_train_cv.add_argument("-s", "--shuffle", dest="shuffle_buffer", type=int, metavar="SHUFFLE_BUFFER_SIZE", default=DEFAULT_SHUFFLE_BUFFER)
    sp_train_cv.add_argument("-b", "--batch", dest="batch_size", type=int, metavar="BATCH_SIZE", default=DEFAULT_BATCH)
    sp_train_cv.add_argument("-e", "--epochs", dest="epochs", type=int, metavar="EPOCHS", default=DEFAULT_EPOCHS)
    sp_train_cv.add_argument("-l", "--smoothing", dest="label_smoothing", type=float, metavar="LABEL_SMOOTHING", default=DEFAULT_LABEL_SMOOTHING)
    sp_train_cv.add_argument("-w", "--weight", dest="onset_positive_weight", type=float, default=DEFAULT_ONSET_POSITIVE_WEIGHT)
    sp_train_cv.add_argument("-i", "--data_base_dir", dest="data_base_dir", default=CQT_PROCESSED_BASE_PATH)
    #sp_train_cv.add_argument("-n", "--no_test", dest="no_test", action="store_true")
    sp_train_cv.add_argument("-v", "--verbosity", dest="verbosity", action="count", default=0)
    sp_train_cv.add_argument("-o", "--output_cv_folder", dest="output_cv_folder_id", default=None)
    sp_train_cv.add_argument("-y", "--dont_save_history", dest="save_history", action="store_false")
    sp_train_cv.add_argument("-g", "--gpu", dest="gpu", default=1)
    sp_train_cv.add_argument("-n", "--num_cv_groups", dest="num_cv_groups", type=int, default=10)
    sp_train_cv.add_argument("-x", "--model_index_to_resume", dest="model_index_to_resume", type=int)


    #sp_evaluate_cv = subparsers.add_parser("evaluate_cv", aliases="ec")



    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    parse_console()