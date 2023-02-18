import os # Temporário, isso vai estar no main.py
import tensorflow as tf
from tensorflow import keras
import glob
import librosa
import matplotlib.pyplot as plt
from librosa.display import specshow
from definitions import *

from model import *
from deserialize_guitarset import prepare_dataset

def restore_model(saved_model_path, label_smoothing=0.2, onset_positive_weight=0.95, lr=0.001):
    model = define_model()
    model.compile(loss=get_loss_dictionary(label_smoothing, onset_positive_weight), 
            optimizer=tf.keras.optimizers.Adam(learning_rate = lr))

    data_path_list = glob.glob(r"guitarset_processed/test/" + r"/*.tfrecord")
    val_dataset = prepare_dataset(data_path_list, buffer_size=10, batch_size=5)

    #model.evaluate(val_dataset, verbose=1)

    model.load_weights(saved_model_path)
    #model.evaluate(val_dataset, verbose=1)

    return model

def plot_examples(saved_model_path):

    model = restore_model(saved_model_path)
    data_path_list = glob.glob(r"guitarset_processed/training/" + r"/*.tfrecord")
    dataset = prepare_dataset(data_path_list, buffer_size=1, batch_size=1)

    for example in dataset.take(1):
        X_contours, X_notes, X_onsets = example[1]["X_contours"][0].numpy().transpose(), example[1]["X_notes"][0].numpy().transpose(), example[1]["X_onsets"][0].numpy().transpose()
        pred = model.predict(example)
        
        pred_X_contours, pred_X_notes, pred_X_onsets = pred["X_contours"][0].transpose(), pred["X_notes"][0].transpose(), pred["X_onsets"][0].transpose()

        plt.figure()
        librosa.display.specshow(X_contours, sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
            hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)
        plt.figure()
        librosa.display.specshow(pred_X_contours, sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
            hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)

        plt.figure()
        librosa.display.specshow(X_notes, sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
            hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)
        plt.figure()
        librosa.display.specshow(pred_X_notes, sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
            hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)

        plt.figure()
        librosa.display.specshow(X_onsets, sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
            hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)
        plt.figure()
        librosa.display.specshow(pred_X_onsets, sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
            hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)
        plt.show()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #restore_model("saved_models/trained.h5")
    plot_examples("saved_models/trained_lr_e3.h5")