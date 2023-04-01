import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import mir_eval
from librosa import midi_to_hz
from model import restore_model_from_weights
from note_creation import model_output_to_notes
from deserialize_guitarset import fetch_dataset, output_batch_to_single
import glob
from definitions import DEFAULT_LEARNING_RATE, DEFAULT_LABEL_SMOOTHING, DEFAULT_ONSET_POSITIVE_WEIGHT, DEFAULT_SHUFFLE_BUFFER, DEFAULT_BATCH

def plot_learning_curves(model_history_path):
    history_dict = json.load(open(f"{model_history_path}", 'r'))
    plt.figure()
    plt.plot(history_dict["loss"])
    plt.plot(history_dict["val_loss"])
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print(f"Training loss: {history_dict['loss'][-1]}")
    print(f"Validation loss: {history_dict['val_loss'][-1]}")

def _convert_to_mir_eval_format(note_creation_output):
    intervals = np.array([[n[0], n[1]] for n in note_creation_output])
    pitch_hz = midi_to_hz(np.array([n[2] for n in note_creation_output]))
    return intervals, pitch_hz

def _get_full_audio_id(segment_audio_id):
    # Get audio identifier from segment identifier. E.g: "xxxx-seg05" -> "xxxx"
    return re.sub(r"-seg\d*$", "", segment_audio_id)

def _mir_evaluate_model_on_files(model, file_path, onset_threshold, frame_threshold):
    # Load data from .tfrecord files on specified path. file_path can be a single path or a list of paths.
    ds = fetch_dataset(file_path)
    full_audio_id = None
    # Iterate through dataset.
    #for example in ds:
    for example in ds: #TODO change to full datset.
        # Unpack example
        id, X_spec, X_contours, X_notes, X_onsets = example
        # Organize target output dictionary.
        target_dict = {"X_contours": X_contours, "X_notes": X_notes, "X_onsets": X_onsets}
        # Get output predictions (same format as target dict).
        predicted_dict = _predict_on_single(model, X_spec)
        # Get audio file identifier (without the segment suffix)
        new_full_audio_id = _get_full_audio_id(id.numpy().decode("ascii"))
        if new_full_audio_id != full_audio_id: # Start of new audio file.       
            # If new file is not the first, process the previous one:
            if full_audio_id is not None:
                print(f"For audio: {full_audio_id}")
                target_intervals, target_pitches = _convert_to_mir_eval_format  (model_output_to_notes(
                            full_audio_target_dict, onset_thresh=0.9, frame_thresh=0.9, create_midi=False
                            ))

                predicted_intervals, predicted_pitches = _convert_to_mir_eval_format(model_output_to_notes(
                            full_audio_predicted_dict, onset_thresh=onset_threshold, frame_thresh=frame_threshold, create_midi=False
                            ))
                
                print("Evaluating:")
                metrics = mir_eval.transcription.evaluate(
                    target_intervals, target_pitches, predicted_intervals, predicted_pitches, offset_ratio=None)
                print(metrics)

            # Start new tensors.
            full_audio_id = new_full_audio_id
            full_audio_target_dict = target_dict
            full_audio_predicted_dict = predicted_dict

        else: # Continuation of previous audio file.        
            # Append tensors. 
            full_audio_target_dict = _concatenate_output_dict(full_audio_target_dict, target_dict, overlap_frames=0)
            full_audio_predicted_dict = _concatenate_output_dict(full_audio_predicted_dict, predicted_dict, overlap_frames=0)
        
        # print("Predicted:")
        # for key, val in full_audio_predicted_dict.items():
        #     print(key, val.shape)
        # print("Target:")
        # for key, val in full_audio_target_dict.items():
        #     print(key, val.shape)
        # print("\n\n")

def _concatenate_output_dict(first_dict, second_dict, overlap_frames):
    """Concatenate first_dict and second_dict tensors along the time axis. Overlap_frames is not yet implemented."""
    return {key: tf.concat([first_tensor, second_dict[key]], axis=0) for (key, first_tensor) in first_dict.items()}


def _predict_on_single(model, X_spec):
    return output_batch_to_single(model.predict_on_batch(np.expand_dims(X_spec, axis=0)))


if __name__ == "__main__":
    _mir_evaluate_model_on_files(
        model=restore_model_from_weights(
            f"saved_models/history_cqt.h5", learning_rate = DEFAULT_LEARNING_RATE, label_smoothing = DEFAULT_LABEL_SMOOTHING,  
            onset_positive_weight = DEFAULT_ONSET_POSITIVE_WEIGHT),
            file_path=["guitarset_processed/training/split_000.tfrecord",
                       "guitarset_processed/training/split_001.tfrecord",
                       "guitarset_processed/training/split_002.tfrecord",
                       "guitarset_processed/training/split_003.tfrecord",
                       "guitarset_processed/training/split_004.tfrecord"
                      ],
            onset_threshold=0.8,
            frame_threshold=0.6
    )


#
#
#
#
# def evaluate_modeeel(model_path, data_base_path, verbose):
#     model = restore_model_from_weights(f"{model_path}.h5", learning_rate = DEFAULT_LEARNING_RATE, label_smoothing = DEFAULT_LABEL_SMOOTHING,  
#                                        onset_positive_weight = DEFAULT_ONSET_POSITIVE_WEIGHT)

#     train_path_list = glob.glob(f"{data_base_path}/training/" + "*.tfrecord")
#     dataset = prepare_dataset(train_path_list, buffer_size = DEFAULT_SHUFFLE_BUFFER, batch_size = DEFAULT_BATCH)

#     val_path_list = glob.glob(f"{data_base_path}/test/" + "*.tfrecord")
#     val_dataset = prepare_dataset(val_path_list, buffer_size = DEFAULT_SHUFFLE_BUFFER, batch_size = DEFAULT_BATCH)

#     print("Training dataset loss function:")
#     model.evaluate(dataset, verbose = verbose)
#     print("Validation dataset loss function:")
#     model.evaluate(val_dataset, verbose = verbose)

