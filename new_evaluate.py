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
from definitions import DEFAULT_LABEL_SMOOTHING, DEFAULT_ONSET_POSITIVE_WEIGHT

def mir_evaluate_model_on_files(model, file_path, onset_threshold, frame_threshold, mode="return"):
    """
    Evaluates MIR metrics for the model on
      the audio files specified in the file path.

    Args:
        model (tensorflow.keras.Model): The trained model to be evaluated.
        file_path (str): The path to the directory containing the audio files to be evaluated.
        onset_threshold (float): The onset threshold for note creation, a value between 0 and 1.
        frame_threshold (float): The frame threshold for note creation, a value between 0 and 1.
        mode (str): The operation mode of the function, either "return" or "print". Defaults to "return".

    Returns:
        If mode == "return", returns a list with the evaluation metrics for each audio file.
        If mode == "print", prints the evaluation metrics for each audio file and returns None.
    """
    assert mode in ("return", "print")

    # Load dataset from given files.
    ds = fetch_dataset(file_path)

    # Initialization.
    full_audio_id = None
    if mode == "return":
        metrics_container = []

    # Iterate through dataset.
    for example in ds:
        # Unpack example
        id, X_spec, X_contours, X_notes, X_onsets = example
        # Organize target output posteriorgrams as dictionary.
        target_dict = {"X_contours": X_contours, "X_notes": X_notes, "X_onsets": X_onsets}
        # Get output posteriorgram predictions (same format as target dict).
        predicted_dict = _predict_on_single(model, X_spec)
        # Get audio identifier (without the segment suffix)
        new_full_audio_id = _get_full_audio_id(id.numpy().decode("ascii"))
        if new_full_audio_id != full_audio_id: # Start of new audio file.       
            # If new audio is not the first, process the previous one, now that its posteriograms are complete:
            if full_audio_id is not None:
                # Evaluate MIR metrics using the target and predicted posteriorgrams.
                metrics = _evaluate_audio(
                            full_audio_target_dict, full_audio_predicted_dict,
                            onset_threshold, frame_threshold
                        )   
                # Print or store metrics, depending on the operation mode.
                if mode == "print":
                    print(f"{full_audio_id}:")
                    print(metrics, end="\n\n")
                else:
                    entry = {"id": full_audio_id}
                    entry.update(metrics)
                    metrics_container.append(entry)

            # Start posteriorgrams for the new audio.
            full_audio_id = new_full_audio_id
            full_audio_target_dict = target_dict
            full_audio_predicted_dict = predicted_dict

        else: # Continuation of previous audio file.

            # Concatenate segments of posteriograms. 
            full_audio_target_dict = _concatenate_output_dict(full_audio_target_dict, target_dict, overlap_frames=0)
            full_audio_predicted_dict = _concatenate_output_dict(full_audio_predicted_dict, predicted_dict, overlap_frames=0)
        
    if mode == "return":
        return metrics_container
    return None

def _convert_to_mir_eval_format(note_creation_output):
    """
    Convert note creation output to the format required by mir_eval transcription evaluation.

    Parameters
    ----------
    note_creation_output : list
        A list of tuples representing note onset, offset, and MIDI pitch.

    Returns
    -------
    tuple
        A tuple of two numpy arrays in the format (intervals, pitch_hz).
        - intervals : numpy.ndarray
            A 2D numpy array of shape (num_notes, 2) representing, for each note, the onset and offset time in seconds.
        - pitch_hz : numpy.ndarray
            A 1D numpy array of length num_notes representing the pitch in Hz for each note.

    """
    intervals = np.array([[n[0], n[1]] for n in note_creation_output])
    pitch_hz = midi_to_hz(np.array([n[2] for n in note_creation_output]))
    return intervals, pitch_hz

def _get_full_audio_id(segment_audio_id):
    """
    Extract the full audio ID from a segmented audio ID.

    Parameters
    ----------
    segment_audio_id : str
        A string representing a segmented audio ID in the format 'audio_id-segN', where N is an integer.

    Returns
    -------
    str
        A string representing the full audio ID without the segment number suffix.

    """
    return re.sub(r"-seg\d*$", "", segment_audio_id)

def _evaluate_audio(target_dict, predicted_dict, onset_threshold, frame_threshold):
    """
    Evaluate transcription metrics for predicted notes compared to target notes.

    Parameters
    ----------
    target_dict : dict
        A dictionary representing the target audio posteriorgrams.
    predicted_dict : dict
        A dictionary representing the predicted audio posteriorgrams.
    onset_threshold : float
        The onset detection threshold used to convert the predicted posteriorgrams to notes.
    frame_threshold : float
        The frame activation threshold used to convert the predicted posteriorgrams to notes.

    Returns
    -------
    dict
        A dictionary of evaluation metrics calculated by mir_eval.transcription.evaluate.
        - 'Onset_F-measure' : float
            The F-measure of the predicted note onsets compared to the target note onsets.
        - 'F-measure' : float
            The F-measure of the predicted note pitches compared to the target note pitches.
        - 'Onset_Precision' : float
            The precision of the predicted note onsets compared to the target note onsets.
        - 'Onset_Recall' : float
            The recall of the predicted note onsets compared to the target note onsets.
        - 'Precision' : float
            The precision of the predicted note pitches compared to the target note pitches.
        - 'Recall' : float
            The recall of the predicted note pitches compared to the target note pitches.

    """
    target_intervals, target_pitches = _convert_to_mir_eval_format  (model_output_to_notes(
                            target_dict, 
                            onset_thresh=0.9, 
                            frame_thresh=0.9, 
                            create_midi=False
                        ))

    predicted_intervals, predicted_pitches = _convert_to_mir_eval_format(model_output_to_notes(
                            predicted_dict, 
                            onset_thresh=onset_threshold, 
                            frame_thresh=frame_threshold, 
                            create_midi=False
                        ))

    return mir_eval.transcription.evaluate(
            target_intervals, target_pitches, predicted_intervals, predicted_pitches, offset_ratio=None)

def _concatenate_output_dict(first_dict, second_dict, overlap_frames=0):
    """
    Concatenate the posteriorgrams from the two dictionaries.

    Parameters
    ----------
    first_dict : dict
        Dictionary of posteriorgrams.
    second_dict : dict
        Another dictionary of posteriorgrams, presumably from the next segment of the same audio.
    overlap_frames : int, optional
        The number of frames to overlap between the tensors in the two dictionaries.
        This parameter is currently not used in this function.

    Returns
    -------
    dict
        A dictionary containing the concatenated posteriorgrams.
        The keys in the returned dictionary are the same as those in first_dict and second_dict.
    """
    return {key: tf.concat([first_tensor, second_dict[key]], axis=0) for (key, first_tensor) in first_dict.items()}

def _predict_on_single(model, X_spec):
    """
    Predict the output for a single input spectrogram.

    Parameters
    ----------
    model : keras.models.Model
        The Keras model used to make the prediction.
    X_spec : numpy.ndarray
        A 2D numpy array representing a single input spectrogram.

    Returns
    -------
    numpy.ndarray
        The model's predicted output dict of posteriorgrams for the given input spectrogram.
    """
    return output_batch_to_single(model.predict_on_batch(np.expand_dims(X_spec, axis=0)))


if __name__ == "__main__":
    dump = mir_evaluate_model_on_files(
            model=restore_model_from_weights(
                "saved_models/fls_base_model/fls_base_model.h5", 
                label_smoothing = DEFAULT_LABEL_SMOOTHING,  
                onset_positive_weight = DEFAULT_ONSET_POSITIVE_WEIGHT
            ),
            file_path=glob.glob("processed/fls_base/train/*.tfrecord"),
            onset_threshold=0.85,
            frame_threshold=0.8
    )


    json.dump(dump, open(f"saved_models/fls_base_model/fls_base_model_predictions_preview.json", 'w'
                         ))
    
    dump = mir_evaluate_model_on_files(
        model=restore_model_from_weights(
            f"saved_models/cqt_base_model/cqt_base_model.h5", 
            label_smoothing = DEFAULT_LABEL_SMOOTHING,  
            onset_positive_weight = DEFAULT_ONSET_POSITIVE_WEIGHT
        ),
        file_path=glob.glob("processed/base/train/*.tfrecord"),
        onset_threshold=0.85,
        frame_threshold=0.8
    )

    json.dump(dump, open(f"saved_models/cqt_base_model/cqt_base_model_preview.json", 'w'
                         ))


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

