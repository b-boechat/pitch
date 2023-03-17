import numpy as np
import matplotlib.pyplot as plt
import mirdata
import librosa
import librosa.display
from math import ceil
import tensorflow as tf
from definitions import *

def _bytes_feature(value):
  """
  Retorna um "bytes_list" de uma string ou byte.
  Função obtida do tutorial do Tensorflow de TFRecord
  """
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def serialize_guitarset_with_splits(splits=[0.9, 0.1], split_names = ["training", "test"], seed=1, 
                                source_dir="guitarset", split_dirs=["dev_guitarset_processed/training", "dev_guitarset_processed/test"]):
    """
    Generate GuitarSet dataset with specified splits and save the processed data to specified directories.

    This function initializes the GuitarSet dataset from the source directory and generates specified splits using a random seed. The processed data of each split is then saved to the corresponding directory specified in split_dirs.

    Args:
    splits (list): A list of proportions for each split, e.g. [0.05, 0.95].
    split_names (list): A list of names for each split, e.g. ["training", "test"].
    seed (int): Seed for generating random splits.
    source_dir (str): Path to the source directory of the GuitarSet dataset.
    split_dirs (list): A list of directories to save the processed data for each split.

    Returns:
    None
    """
    assert(len(splits) == len(split_names) and len(splits) == len(split_dirs))
    
    # Inicializa dataset "guitarset" do diretório raiz.    
    data = mirdata.initialize("guitarset", data_home=source_dir)

    # Obtém splits definidos (por padrão, treinamento e teste).
    splits_dict = data.get_random_track_splits(splits=splits, seed=seed, split_names=split_names)
    tracks = data.load_tracks()
    
    # Processa cada split.
    for i, name in enumerate(split_names):
        print(f"Processing split: \"{name}\"...")
        process_split(tracks, splits_dict[name], split_dirs[i])
    
    print("Done.")
    
    # Debug: processa só o split de treinamento, com [0.02]
    #process_split(tracks, split_keys_list=splits_dict["training"], splits_dir="dev_guitarset_processed/training")
    
def process_split(tracks, split_keys_list, splits_dir):
    """This function takes in a dictionary of audio tracks, a list of keys that correspond to the tracks, and the directory 
    where the splits are stored. It splits the audio tracks into smaller pieces, processes each piece to get the 
    spectrogram, and saves the results in TensorFlow record files.

    tracks: a dictionary containing audio tracks as values and the keys that correspond to each track.
    split_keys_list: a list of keys that correspond to the tracks in the 'tracks' dictionary.
    splits_dir: the directory where the splits are stored.
    """
    # Get the number of audio tracks and calculate the number of record files needed.
    num_tracks = len(split_keys_list)
    num_record_files = int(ceil(num_tracks / NUM_TRACKS_PER_RECORD_FILE))

    # Split the audio tracks into smaller pieces and save them in TensorFlow record files.
    for i in range(num_record_files - 1):
        # Create a TensorFlow record writer for the current record file.
        with tf.io.TFRecordWriter(f"{splits_dir}/split_{i:03d}.tfrecord") as writer:
            print(f"Writing \"{splits_dir}/split_{i:03d}.tfrecord\"")
            for j in range(NUM_TRACKS_PER_RECORD_FILE):
                # Get the index of the current audio track.
                track_index = i * NUM_TRACKS_PER_RECORD_FILE + j
                # Get the key that corresponds to the current audio track.
                key = split_keys_list[track_index]
                # Process the audio track to get the spectrogram and serialize the results.
                serialized_track_examples = process_track(key, tracks[key])
                # Write the serialized results to the current TensorFlow record file.
                for example in serialized_track_examples:
                    writer.write(example)

    # Do the same for the last record file, which might have fewer audio tracks than the others.
    with tf.io.TFRecordWriter(f"{splits_dir}/split_{num_record_files-1:03d}.tfrecord") as writer:
        print(f"Writing \"{splits_dir}/split_{num_record_files-1:03d}.tfrecord\"")
        for track_index in range((num_record_files - 1) * NUM_TRACKS_PER_RECORD_FILE, num_tracks):
            key = split_keys_list[track_index]
            serialized_track_examples = process_track(key, tracks[key])
            for example in serialized_track_examples:
                writer.write(example)
    print("=======")



def process_track(key, track):
    """Process the audio files and compute the CQT spectrogram for each split"""

    audio_data, _ = librosa.load(track.audio_mic_path, sr=AUDIO_SAMPLE_RATE)
    
    # Calcula o espectrograma em dB de CQT para o arquivo de áudio. Ele tem dimensão FREQUÊNCIA X TEMPO.
    X_spec = get_cqt_spectrogram(audio_data)

    # Calcula o número de segmentos contidos no áudio. 
    num_segments = int(ceil(X_spec.shape[0] / AUDIO_SEGMENT_LEN_FRAMES))
    
    # Realiza zero-padding no espectrograma para que todos os segmentos tenham o mesmo tamanho.
    time_padding = num_segments * AUDIO_SEGMENT_LEN_FRAMES - X_spec.shape[0]
    X_spec = np.pad(X_spec, ((0, time_padding), (0, 0)))
    
    # Obtém as escalas de tempo e frequência para as anotações.
    annotations_time_scale = librosa.frames_to_time(
        np.arange(X_spec.shape[0]), sr=AUDIO_SAMPLE_RATE, hop_length=CQT_HOP_LENGTH)
    contours_freq_scale = librosa.cqt_frequencies(
        n_bins=CONTOURS_TOTAL_BINS, fmin = MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)
    notes_freq_scale = librosa.cqt_frequencies(
        n_bins=NOTES_TOTAL_BINS, fmin = MINIMUM_ANNOTATION_FREQUENCY, bins_per_octave=NOTES_BINS_PER_OCTAVE, tuning=0.0)

    # Obtém as matrizes com as anotações. Elas estão em dimensão TEMPO X FREQUÊNCIA.
    X_contours = track.multif0.to_matrix(annotations_time_scale, "s", contours_freq_scale, "hz", amplitude_unit="binary")
    X_notes = track.notes_all.to_matrix(annotations_time_scale, "s", notes_freq_scale, "hz", amplitude_unit="binary", onsets_only=False)
    X_onsets = track.notes_all.to_matrix(annotations_time_scale, "s", notes_freq_scale, "hz", amplitude_unit="binary", onsets_only=True)

    # Suprime o warning que seria dado pelo numpy ao obter as matrizes de anotações. (não funcionando)
    # np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # Divide o espectrograma e as matrizes de anotações em segmentos, pelo eixo do tempo.
    X_spec_segments = np.split(X_spec, num_segments, axis=0)
    X_contours_segments = np.split(X_contours, num_segments, axis=0)
    X_notes_segments = np.split(X_notes, num_segments, axis=0)
    X_onsets_segments = np.split(X_onsets, num_segments, axis=0)

    #print(f"Complete padded CQT shape: {X_spec.shape}")
    #print(f"Spec segs shape: {np.array(X_spec_segments).shape}")
    #print(f"Contours segs shape: {np.array(X_contours_segments).shape}")
    #print(f"Notes segs shape: {np.array(X_notes_segments).shape}")
    #print(f"Onsets segs shape: {np.array(X_onsets_segments).shape}")

    serialized_examples_list = []

    for i, X_spec_seg in enumerate(X_spec_segments):
        # Obtém os índices para as representações esparsas
        X_contours_seg_sparse = tf.sparse.from_dense(X_contours_segments[i]).indices
        if tf.size(X_contours_seg_sparse) == 0:
            continue
        X_notes_seg_sparse = tf.sparse.from_dense(X_notes_segments[i]).indices
        X_onsets_seg_sparse = tf.sparse.from_dense(X_onsets_segments[i]).indices

        contours_len = X_contours_seg_sparse.shape[0]
        notes_len = X_notes_seg_sparse.shape[0]
        onsets_len = X_onsets_seg_sparse.shape[0]

        identifier = f"{key}-seg{i:02d}"
 
        #print("Spec", tf.convert_to_tensor(X_spec_seg))
        #print("Contours", X_contours_seg_sparse)
        #print("Notes", X_notes_seg_sparse)

        # id: dtype = tf.String
        # Spec: dtype = tf.Float32.
        # Contours: dtype = tf.Int64.
        # Contours_len : dtype = tf.Int64
        # Notes: dtype = tf.Int64
        # Notes_len : dtype = tf.Int64
        # Onsets: dtype = tf.Int64
        # Onsets_len : dtype = tf.Int64
        serialized_examples_list.append(serialize_example_segment(
            identifier, 
            X_spec_seg, 
            X_contours_seg_sparse, contours_len,
            X_notes_seg_sparse, notes_len,
            X_onsets_seg_sparse, onsets_len))

    return serialized_examples_list


            
def serialize_example_segment(identifier, X_spec_seg, X_contours_seg, contours_len, X_notes_seg, notes_len, X_onsets_seg, onsets_len):
    ser_X_spec_seg = tf.io.serialize_tensor(X_spec_seg)
    ser_X_contours_seg = tf.io.serialize_tensor(X_contours_seg)
    ser_X_notes_seg = tf.io.serialize_tensor(X_notes_seg)
    ser_X_onsets_seg = tf.io.serialize_tensor(X_onsets_seg)

    feature_dict = {
        "id": _bytes_feature(identifier.encode("ascii")),
        "X_spec": _bytes_feature(ser_X_spec_seg),
        "X_contours": _bytes_feature(ser_X_contours_seg),
        "contours_len": _int64_feature(contours_len),
        "X_notes": _bytes_feature(ser_X_notes_seg),
        "notes_len": _int64_feature(notes_len),
        "X_onsets": _bytes_feature(ser_X_onsets_seg),
        "onsets_len": _int64_feature(onsets_len)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()


def get_cqt_spectrogram(audio_data):
    # Calcula o espectrograma em dB de CQT para o arquivo de áudio.
    X_spec = librosa.amplitude_to_db(np.abs(librosa.cqt(audio_data, sr=AUDIO_SAMPLE_RATE, 
                            hop_length=CQT_HOP_LENGTH, fmin=MINIMUM_ANNOTATION_FREQUENCY, 
                            n_bins=CQT_TOTAL_BINS, bins_per_octave=CONTOURS_BINS_PER_OCTAVE, tuning=0.0)))
        
    # Transpõe o espectrograma para TEMPO X FREQUÊNCIA, adequando-o às anotações e ao modelo.
    return X_spec.transpose()


def generate_file_name(key, segment, representation):
    pass

# https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
# https://www.tensorflow.org/tutorials/load_data/tfrecord

if __name__ == "__main__":
    serialize_guitarset_with_splits()