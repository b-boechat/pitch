import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import librosa
import numpy as np

import json
import mirdata
from uuid import uuid4
from math import ceil
from definitions import NUM_TRACKS_PER_RECORD_FILE, AUDIO_SAMPLE_RATE, AUDIO_SEGMENT_LEN_FRAMES, AUDIO_SEGMENT_LEN_SECS, \
                        CQT_HOP_LENGTH, CONTOURS_TOTAL_BINS, MINIMUM_ANNOTATION_FREQUENCY, \
                        CONTOURS_BINS_PER_OCTAVE, NOTES_TOTAL_BINS, NOTES_BINS_PER_OCTAVE , \
                        CQT_TOTAL_BINS, RAW_DATASETS_BASE_PATH, PROCESSED_DATASETS_BASE_PATH
from swgm import swgm_cython_wrapper
from fls import fls_cython_wrapper


class GuitarsetSerializer:
    def __init__(self, destination_base_dir_name="test", 
                 source_dir_name="guitarset",
                 num_tracks_per_record_file=NUM_TRACKS_PER_RECORD_FILE,
                 audio_sample_rate=AUDIO_SAMPLE_RATE, 
                 audio_segment_len_frames=AUDIO_SEGMENT_LEN_FRAMES, 
                 cqt_hop_length=CQT_HOP_LENGTH, 
                 cqt_total_bins=CQT_TOTAL_BINS,
                 contours_total_bins=CONTOURS_TOTAL_BINS, 
                 minimum_annotation_frequency=MINIMUM_ANNOTATION_FREQUENCY, 
                 contours_bins_per_octave=CONTOURS_BINS_PER_OCTAVE, 
                 notes_total_bins=NOTES_TOTAL_BINS, 
                 notes_bins_per_octave=NOTES_BINS_PER_OCTAVE, 
                 combination_method_str=None,
                 combination_params={},
                 ) -> None:
        """
        destination_base_dir_name (string): Name of the base destination directory for the processed dataset. Implied path is "PROCESSED_DATASETS_BASE_PATH/destination_base_dir_name". Subdirectories for each split are automatically created.
        Continue docstring.
        """
        self.destination_base_dir_name = destination_base_dir_name
        self.source_dir_name = source_dir_name
        self.num_tracks_per_record_file = num_tracks_per_record_file
        self.audio_sample_rate = audio_sample_rate
        self.audio_segment_len_frames = audio_segment_len_frames
        self.cqt_hop_length = cqt_hop_length
        self.cqt_total_bins = cqt_total_bins
        self.contours_total_bins = contours_total_bins
        self.minimum_annotation_frequency = minimum_annotation_frequency
        self.contours_bins_per_octave = contours_bins_per_octave
        self.notes_total_bins = notes_total_bins
        self.notes_bins_per_octave = notes_bins_per_octave
        if combination_method_str is not None:
            assert combination_method_str in ("swgm", "fls")
        self.combination_method_str = combination_method_str
        self.combination_params = combination_params
    
    def _get_spectrogram(self, audio_data, filter_scales=[1/3, 2/3, 1], **kwargs):
        if self.combination_method_str is None:
            return self._get_cqt_spectrogram(audio_data).transpose()
        specs_tensor = np.array([librosa.cqt(audio_data, sr=self.audio_sample_rate, 
                                hop_length=self.cqt_hop_length, fmin=self.minimum_annotation_frequency, 
                                n_bins=self.cqt_total_bins, bins_per_octave=self.contours_bins_per_octave, tuning=0.0,
                                filter_scale=scale) for scale in filter_scales])
        audio_energy = np.linalg.norm(audio_data)
        specs_tensor = np.square(np.abs(specs_tensor)).astype(np.double)

        specs_tensor *= audio_energy / np.linalg.norm(specs_tensor, axis=(1, 2), keepdims=True)

        if self.combination_method_str == "swgm":
            spec = swgm_cython_wrapper(specs_tensor, **kwargs)
        else: # fls
            spec = fls_cython_wrapper(specs_tensor, **kwargs)
        spec = np.float32(spec)
        spec *= audio_energy/np.sum(spec, axis=None)
        return librosa.power_to_db(spec).transpose()


    def _get_cqt_spectrogram(self, audio_data):
        # Calculate dB CQT spectrogram for audio data.
        return librosa.amplitude_to_db(np.abs(librosa.cqt(audio_data, sr=self.audio_sample_rate, 
                                hop_length=self.cqt_hop_length, fmin=self.minimum_annotation_frequency, 
                                n_bins=self.cqt_total_bins, bins_per_octave=self.contours_bins_per_octave, tuning=0.0)))



    def serialize(self, splits=[0.8, 0.1, 0.1], split_names = ["train", "val", "test"], seed=1):
        """
        Generate GuitarSet dataset with specified splits and save the processed data to specified directories.

        This function initializes the GuitarSet dataset from the source directory and generates specified splits using a random seed. The processed data of each split is then saved to the corresponding directory specified in split_dirs.

        Args:
        splits (list): A list of proportions for each split, e.g. [0.05, 0.95].
        split_names (list): A list of names for each split, e.g. ["training", "test"].
        seed (int): Seed for generating random splits.
        source_dir (str): Path to the source directory of the GuitarSet dataset.

        Returns:
        None
        """

        assert(len(splits) == len(split_names))
        
        # Initialize guitarset from source.
        source_path = f"{RAW_DATASETS_BASE_PATH}/{self.source_dir_name}"
        data = mirdata.initialize("guitarset", data_home=source_path)

        # Get splits using helper mirdata function.
        splits_dict = data.get_random_track_splits(splits=splits, seed=seed, split_names=split_names)
        tracks = data.load_tracks()

        # Create destination base directory.
        destination_base_path = f"{PROCESSED_DATASETS_BASE_PATH}/{self.destination_base_dir_name}" 
        if os.path.isdir(destination_base_path):
            print(f"Destination '{destination_base_path}' exists.")
            destination_base_path = f"{destination_base_path}-{uuid4().hex}"
        os.mkdir(destination_base_path)
        
        print(f"Writing to '{destination_base_path}'.")
        
        # Write metadata.
        metadata = {
            "source_dir": self.source_dir_name,
            "num_tracks_per_record_file": self.num_tracks_per_record_file,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_segment_len_frames": self.audio_segment_len_frames,
            "cqt_hop_length": self.cqt_hop_length,
            "cqt_total_bins": self.cqt_total_bins,
            "contours_total_bins": self.contours_total_bins,
            "minimum_annotation_frequency": self.minimum_annotation_frequency,
            "contours_bins_per_octave": self.contours_bins_per_octave,
            "notes_total_bins": self.notes_total_bins,
            "notes_bins_per_octave": self.notes_bins_per_octave,
            "combination_method_str": self.combination_method_str,
            "combination_params": self.combination_params,
            "splits": splits,
            "split_names": split_names
        }
        json.dump(metadata, open(f"{destination_base_path}/{self.destination_base_dir_name}_metadata.json", 'w'))

        # Process each split.
        for name in split_names:
            print(f"Processing split: \"{name}\"...")
            split_path = f"{destination_base_path}/{name}"
            os.mkdir(split_path)
            self._process_split(tracks, splits_dict[name], split_path)
        
        print("Done.")

    def _process_split(self, tracks, split_keys_list, splits_dir):
        """This function takes in a dictionary of audio tracks, a list of keys that correspond to the tracks, and the directory 
        where the splits are stored. It splits the audio tracks into smaller pieces, processes each piece to get the 
        spectrogram, and saves the results in TensorFlow record files.

        tracks: a dictionary containing audio tracks as values and the keys that correspond to each track.
        split_keys_list: a list of keys that correspond to the tracks in the 'tracks' dictionary.
        splits_dir: the directory where the splits are stored.
        """
        # Get the number of audio tracks and calculate the number of record files needed.
        num_tracks = len(split_keys_list)
        num_record_files = int(ceil(num_tracks / self.num_tracks_per_record_file))

        # Split the audio tracks into smaller pieces and save them in TensorFlow record files.
        for i in range(num_record_files - 1):
            # Create a TensorFlow record writer for the current record file.
            with tf.io.TFRecordWriter(f"{splits_dir}/file_{i:03d}.tfrecord") as writer:
                print(f"Writing \"{splits_dir}/file_{i:03d}.tfrecord\"")
                for j in range(self.num_tracks_per_record_file):
                    # Get the index of the current audio track.
                    track_index = i * self.num_tracks_per_record_file + j
                    # Get the key that corresponds to the current audio track.
                    key = split_keys_list[track_index]
                    # Process the audio track to get the spectrogram and serialize the results.
                    serialized_track_examples = self._process_track(key, tracks[key])
                    # Write the serialized results to the current TensorFlow record file.
                    for example in serialized_track_examples:
                        writer.write(example)

        # Do the same for the last record file, which might have fewer audio tracks than the others.
        with tf.io.TFRecordWriter(f"{splits_dir}/file_{num_record_files-1:03d}.tfrecord") as writer:
            print(f"Writing \"{splits_dir}/file_{num_record_files-1:03d}.tfrecord\"")
            for track_index in range((num_record_files - 1) * self.num_tracks_per_record_file, num_tracks):
                key = split_keys_list[track_index]
                serialized_track_examples = self._process_track(key, tracks[key])
                for example in serialized_track_examples:
                    writer.write(example)
        print("=======")

    #AUDIO_SAMPLE_RATE, AUDIO_SEGMENT_LEN_FRAMES, CQT_HOP_LENGTH, CONTOURS_TOTAL_BINS, MINIMUM_ANNOTATION_FREQUENCY, CONTOURS_BINS_PER_OCTAVE, NOTES_TOTAL_BINS, NOTES_BINS_PER_OCTAVE

    def _process_track(self, key, track):
        """Process the audio files and compute the CQT spectrogram for each split"""

        audio_data, _ = librosa.load(track.audio_mic_path, sr=self.audio_sample_rate)
        
        # Calcula o espectrograma em dB para o arquivo de áudio. Ele tem dimensão TEMPO X FREQUÊNCIA, e pode ser uma CQT ou uma representação combinada implementada.
        X_spec = self._get_spectrogram(audio_data, **self.combination_params)

        # Calcula o número de segmentos contidos no áudio. 
        num_segments = int(ceil(X_spec.shape[0] / self.audio_segment_len_frames))
        
        # Realiza zero-padding no espectrograma para que todos os segmentos tenham o mesmo tamanho.
        time_padding = num_segments * self.audio_segment_len_frames - X_spec.shape[0]
        X_spec = np.pad(X_spec, ((0, time_padding), (0, 0)))
        
        # Obtém as escalas de tempo e frequência para as anotações.
        annotations_time_scale = librosa.frames_to_time(
            np.arange(X_spec.shape[0]), sr=self.audio_sample_rate, hop_length=self.cqt_hop_length)
        contours_freq_scale = librosa.cqt_frequencies(
            n_bins=self.contours_total_bins, fmin = self.minimum_annotation_frequency, bins_per_octave=self.contours_bins_per_octave, tuning=0.0)
        notes_freq_scale = librosa.cqt_frequencies(
            n_bins=self.notes_total_bins, fmin = self.minimum_annotation_frequency, bins_per_octave=self.notes_bins_per_octave, tuning=0.0)

        # Obtém as matrizes com as anotações. Elas estão em dimensão TEMPO X FREQUÊNCIA.
        X_contours = track.multif0.to_matrix(annotations_time_scale, "s", contours_freq_scale, "hz", amplitude_unit="binary")
        X_notes = track.notes_all.to_matrix(annotations_time_scale, "s", notes_freq_scale, "hz", amplitude_unit="binary", onsets_only=False)
        X_onsets = track.notes_all.to_matrix(annotations_time_scale, "s", notes_freq_scale, "hz", amplitude_unit="binary", onsets_only=True)

        # Divide o espectrograma e as matrizes de anotações em segmentos, pelo eixo do tempo.
        X_spec_segments = np.split(X_spec, num_segments, axis=0)
        X_contours_segments = np.split(X_contours, num_segments, axis=0)
        X_notes_segments = np.split(X_notes, num_segments, axis=0)
        X_onsets_segments = np.split(X_onsets, num_segments, axis=0)

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

            serialized_examples_list.append(self._serialize_example_segment(
                identifier, 
                X_spec_seg, 
                X_contours_seg_sparse, contours_len,
                X_notes_seg_sparse, notes_len,
                X_onsets_seg_sparse, onsets_len))

        return serialized_examples_list
    
                
    def _serialize_example_segment(self, identifier, X_spec_seg, X_contours_seg, contours_len, X_notes_seg, notes_len, X_onsets_seg, onsets_len):
        ser_X_spec_seg = tf.io.serialize_tensor(X_spec_seg)
        ser_X_contours_seg = tf.io.serialize_tensor(X_contours_seg)
        ser_X_notes_seg = tf.io.serialize_tensor(X_notes_seg)
        ser_X_onsets_seg = tf.io.serialize_tensor(X_onsets_seg)

        feature_dict = {
            "id": self._bytes_feature(identifier.encode("ascii")),
            "X_spec": self._bytes_feature(ser_X_spec_seg),
            "X_contours": self._bytes_feature(ser_X_contours_seg),
            "contours_len": self._int64_feature(contours_len),
            "X_notes": self._bytes_feature(ser_X_notes_seg),
            "notes_len": self._int64_feature(notes_len),
            "X_onsets": self._bytes_feature(ser_X_onsets_seg),
            "onsets_len": self._int64_feature(onsets_len)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()

    def show_size_information(self, splits=[0.8, 0.1, 0.1], split_names = ["train", "val", "test"], seed=1):
        # Initialize guitarset from source.
        source_path = f"{RAW_DATASETS_BASE_PATH}/{self.source_dir_name}"
        data = mirdata.initialize("guitarset", data_home=source_path)

        # Get splits using helper mirdata function.
        splits_dict = data.get_random_track_splits(splits=splits, seed=seed, split_names=split_names)
        tracks = data.load_tracks()

        # TODO change this if overlap is included.
        total_dataset_duration = 0.0
        for split, track_keys in splits_dict.items():
            total_split_duration = 0.0
            total_segs = 0
            for key in track_keys:
                duration = librosa.get_duration(filename=tracks[key].audio_mic_path)
                segs = ceil(duration / AUDIO_SEGMENT_LEN_SECS)
                total_split_duration += duration
                total_segs += segs
            total_dataset_duration += total_split_duration
            print(f"Split: {split} \nFiles = {len(track_keys)}\nDuration = {total_split_duration//60} min {total_split_duration%60} s\nSegs = {total_segs}", end="\n\n")
        print(f"Total duration: {total_dataset_duration//60} min {total_dataset_duration%60} s")




    # Funções obtidas do tutorial do Tensorflow de TFRecord.
    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    

if __name__ == "__main__":
    #serializer = GuitarsetSerializer()
    #serializer = GuitarsetSerializer(destination_base_dir_name="fls_base", combination_method_str="fls")
    #serializer = GuitarsetSerializer(destination_base_dir_name="fls_fw11", combination_method_str="fls", combination_params={'freq_width': 11})
    #serializer = GuitarsetSerializer(destination_base_dir_name="swgm_base", combination_method_str="swgm")
    serializer_cqt = GuitarsetSerializer(
            destination_base_dir_name="cqt_all",
            combination_method_str=None
        )
    serializer_cqt.serialize(splits=[0.9, 0.1], split_names=["cv", "test"], seed=1)

    serializer_swgm = GuitarsetSerializer(
            destination_base_dir_name="swgm_all", 
            combination_method_str="swgm"
        )
    serializer_swgm.serialize(splits=[0.9, 0.1], split_names=["cv", "test"], seed=1)

    serializer_fls = GuitarsetSerializer(
        destination_base_dir_name="fls_fw11", 
        combination_method_str="fls", 
        combination_params={'freq_width': 11}
    )
    serializer_fls.serialize(splits=[0.9, 0.1], split_names=["cv", "test"], seed=1)