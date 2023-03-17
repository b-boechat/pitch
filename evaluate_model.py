import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import glob
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt
from model import restore_model_from_weights
from deserialize_guitarset import prepare_dataset
from definitions import DEFAULT_LEARNING_RATE, DEFAULT_LABEL_SMOOTHING, DEFAULT_ONSET_POSITIVE_WEIGHT, \
    CQT_HOP_LENGTH, MINIMUM_ANNOTATION_FREQUENCY, CONTOURS_BINS_PER_OCTAVE, NOTES_BINS_PER_OCTAVE, AUDIO_SAMPLE_RATE

def evaluate_model(model_path):
    model = restore_model_from_weights(model_path, learning_rate = DEFAULT_LEARNING_RATE, label_smoothing = DEFAULT_LABEL_SMOOTHING,  
                                       onset_positive_weight = DEFAULT_ONSET_POSITIVE_WEIGHT)
    data_path_list = glob.glob(r"guitarset_processed/training/" + r"/*.tfrecord")
    dataset = prepare_dataset(data_path_list, buffer_size = 5, batch_size = 32)

    val_path_list = glob.glob(r"guitarset_processed/test/" + r"/*.tfrecord")
    val_dataset = prepare_dataset(val_path_list, buffer_size = 5, batch_size = 32)

    print("Train:")
    model.evaluate(dataset, verbose = 2)
    print("Validation:")
    model.evaluate(val_dataset, verbose = 2)


if __name__ == "__main__":
    evaluate_model("saved_models/trained_lr_e3.h5")
