import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import librosa
#import librosa.display
from definitions import AUDIO_SEGMENT_LEN_FRAMES, NOTES_TOTAL_BINS, CONTOURS_TOTAL_BINS

from definitions import *

def prepare_dataset(filenames, buffer_size, batch_size, num_parallel_reads=None):
    """Fetch and prepare a dataset for training or evaluation.
    """
    dataset = read_raw_dataset_from_files(filenames, num_parallel_reads)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(_deserialize_example)
    dataset = dataset.map(zip_for_model)
    dataset = dataset.batch(batch_size, drop_remainder=True)#.prefetch(tf.data.AUTOTUNE)
    return dataset

def fetch_dataset(filenames, num_parallel_reads=None):
    """Fetch a dataset without shuffling, batching or zipping.
    """
    dataset = read_raw_dataset_from_files(filenames, num_parallel_reads)
    dataset = dataset.map(_deserialize_example)
    return dataset

def zip_for_model(_, X_spec, X_contours, X_notes, X_onsets):
    """Transform an example (as returned by deserialize example) into a tuple (inputs, outputs), expected by model.fit.
    """
    return X_spec, {
        "X_contours": X_contours, 
        "X_notes" : X_notes,
        "X_onsets" : X_onsets
    }

def read_raw_dataset_from_files(filenames, num_parallel_reads=None):
    return tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel_reads)

def _deserialize_example(example_proto):
    """Deserialize a single example from a TFRecord file.

    The input `example_proto` is a serialized example from a TFRecord file.
    This function deserializes the example, parses its features and returns them.

    Args:
        example_proto (tf.train.Example): a serialized example from a TFRecord file.

    Returns:
        tuple: a tuple of five elements, containing the deserialized features:
            id (tf.Tensor): a tensor of shape `[1]` and type `tf.string`, containing the example identifier.
            X_spec (tf.Tensor): a tensor of shape `[AUDIO_SEGMENT_LEN_FRAMES, SPECTROGRAM_BINS]` and type `tf.float32`, containing the spectrogram features.
            X_contours (tf.Tensor): a tensor of shape `[AUDIO_SEGMENT_LEN_FRAMES, CONTOURS_TOTAL_BINS]` and type `tf.float32`, containing the contours features.
            X_notes (tf.Tensor): a tensor of shape `[AUDIO_SEGMENT_LEN_FRAMES, NOTES_TOTAL_BINS]` and type `tf.float32`, containing the notes features.
            X_onsets (tf.Tensor): a tensor of shape `[AUDIO_SEGMENT_LEN_FRAMES, NOTES_TOTAL_BINS]` and type `tf.float32`, containing the onsets features.
    """
    # Parse the example_proto into a dictionary of features.
    parsed_example = tf.io.parse_single_example(example_proto, {
        "id": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "X_spec": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "X_contours": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "contours_len": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "X_notes": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "notes_len": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "X_onsets": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "onsets_len": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    } )

    # Get the identifier of the example.
    id = parsed_example['id']

    # Get the spectrogram features of the example.
    X_spec = tf.io.parse_tensor(parsed_example['X_spec'], out_type=tf.float32)
    
    # Get the serialized annotation matrixes, in sparse form (coordinates).
    contours_coords = tf.io.parse_tensor(parsed_example['X_contours'], out_type=tf.int64)
    notes_coords = tf.io.parse_tensor(parsed_example['X_notes'], out_type=tf.int64)
    onsets_coords = tf.io.parse_tensor(parsed_example['X_onsets'], out_type=tf.int64)
    
    # Get dense forms for the annotation matrixes.
    X_contours = tf.sparse.to_dense(tf.sparse.SparseTensor(contours_coords, tf.ones(parsed_example['contours_len'], dtype=tf.float32), [AUDIO_SEGMENT_LEN_FRAMES, CONTOURS_TOTAL_BINS]))
    X_notes = tf.sparse.to_dense(tf.sparse.SparseTensor(notes_coords, tf.ones(parsed_example['notes_len'], dtype=tf.float32), [AUDIO_SEGMENT_LEN_FRAMES, NOTES_TOTAL_BINS]))
    X_onsets = tf.sparse.to_dense(tf.sparse.SparseTensor(onsets_coords, tf.ones(parsed_example['onsets_len'], dtype=tf.float32), [AUDIO_SEGMENT_LEN_FRAMES, NOTES_TOTAL_BINS]))

    return id, X_spec, X_contours, X_notes, X_onsets

def zipped_single_to_batch(single):
    """Transform a single zipped example into a batch of size 1. Useful for doing predictions on a single element.
    """
    return np.expand_dims(single[0], axis=0), {
        "X_contours": np.expand_dims(single[1]["X_contours"], axis=0), 
        "X_notes": np.expand_dims(single[1]["X_notes"], axis=0), 
        "X_onsets": np.expand_dims(single[1]["X_onsets"], axis=0), 
    }

def output_batch_to_single(batch):
    """Transform a batch output of size 1 into a single output dictionary. Useful for accessing predicted features.
    """
    return {
        "X_contours": batch["X_contours"][0], 
        "X_notes": batch["X_notes"][0], 
        "X_onsets": batch["X_onsets"][0], 
    }


if __name__ == "__main__":
    dataset = read_raw_dataset_from_files("guitarset_processed/training/split_002.tfrecord")
    for example_proto in dataset.take(1):
        print(_deserialize_example(example_proto))
        #_visualize_example(example)