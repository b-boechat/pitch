# OUTDATED FILE!

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import glob
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt
from model import restore_model_from_weights
from definitions import DEFAULT_LEARNING_RATE, DEFAULT_LABEL_SMOOTHING, DEFAULT_ONSET_POSITIVE_WEIGHT, \
    CQT_HOP_LENGTH, MINIMUM_ANNOTATION_FREQUENCY, CONTOURS_BINS_PER_OCTAVE, NOTES_BINS_PER_OCTAVE, AUDIO_SAMPLE_RATE
from deserialize_guitarset import fetch_dataset, zip_for_model, zipped_single_to_batch, output_batch_to_single

def visualize_predictions(saved_model_path, data_path_list):
    model = restore_model_from_weights(saved_model_path, learning_rate = DEFAULT_LEARNING_RATE, label_smoothing = DEFAULT_LABEL_SMOOTHING,  
                                       onset_positive_weight = DEFAULT_ONSET_POSITIVE_WEIGHT)

    dataset = fetch_dataset(data_path_list)

    for example in dataset:
        id, X_spec, X_contours, X_notes, X_onsets = example

        pred = output_batch_to_single(
                model.predict_on_batch(
                    zipped_single_to_batch( 
                        zip_for_model(*example))))

        plt.figure()
        specshow(X_spec.numpy().transpose(), sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
                 hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, tuning=0.0,bins_per_octave=CONTOURS_BINS_PER_OCTAVE,
                 )

        fig, axs = plt.subplots(2, 2)
        # ======== Plot contours
        specshow(X_contours.numpy().transpose(), sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
                 hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, tuning=0.0,bins_per_octave=CONTOURS_BINS_PER_OCTAVE,
                 ax=axs[0][0]
                 )
        specshow(pred["X_contours"].transpose(), sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
                 hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, tuning=0.0,bins_per_octave=CONTOURS_BINS_PER_OCTAVE,
                 ax=axs[1][0]
                 )
        # ======== Plot notes
        specshow(X_notes.numpy().transpose(), sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
                 hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, tuning=0.0,bins_per_octave=NOTES_BINS_PER_OCTAVE,
                 ax=axs[0][1]
                 )
        specshow(pred["X_notes"].transpose(), sr=AUDIO_SAMPLE_RATE, x_axis='time', y_axis='cqt_hz',
                 hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, tuning=0.0,bins_per_octave=NOTES_BINS_PER_OCTAVE,
                 ax=axs[1][1]
                 )
        plt.subplots_adjust(bottom=0.02, left=0.02, right=0.98, top=0.98, wspace=0.1, hspace=0.1)
        print(f"Id: {id}")
        #plt.get_current_fig_manager().full_screen_toggle()
        plt.show()
        if input(".") == "x":
            exit()


if __name__ == "__main__":
    visualize_predictions("saved_models/history_swgm_e200.h5", glob.glob(r"guitarset_swgm/training" + r"/*.tfrecord"))
    
                        
    
