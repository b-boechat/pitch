import argparse as ap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from definitions import DEFAULT_BATCH, DEFAULT_ONSET_POSITIVE_WEIGHT, \
    DEFAULT_SHUFFLE_BUFFER, DEFAULT_LEARNING_RATE, DEFAULT_LABEL_SMOOTHING, \
    CQT_PROCESSED_BASE_PATH, DEFAULT_EPOCHS
from train import train
from read_evaluation import read_metrics

def train_wrapper(args):
    if args.verbosity > 2:
        args.verbosity = 2
    elif args.verbosity < 0:
        args.verbosity = 0

    print(args)

    with tf.device('/GPU:1'):
        train(learning_rate=args.learning_rate,
                label_smoothing=args.label_smoothing,
                buffer_size=args.shuffle_buffer,
                batch_size=args.batch_size,
                epochs=args.epochs,
                onset_positive_weight=args.onset_positive_weight,
                verbose=args.verbosity,
                data_base_dir=args.data_base_dir,
                output_folder_id=args.output_folder_id,
                save_history=args.save_history
        )

def read_metrics_wrapper(args): # TODO add other arguments
    read_metrics(args.model_id)

def parse_console():
    parser = ap.ArgumentParser(description="Interface para treinamento e avaliação do sistema de detecção de frequência fundamental baseado no \"basic pitch\"")

    subparsers = parser.add_subparsers()

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


    sp_read_eval = subparsers.add_parser("read_eval", aliases="r")
    sp_read_eval.set_defaults(func=read_metrics_wrapper)
    sp_read_eval.add_argument("model_id")

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    parse_console()