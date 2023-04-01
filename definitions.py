from math import floor, ceil, log2

RANDOM_SEED = 1
GUITARSET_BASE_PATH = "D:\\mirdata datasets\guitarset"
GUITARSET_PROCESSED_BASE_PATH = "guitarset_cqt"
AUDIO_SAMPLE_RATE = 22050 # A referência usa 22050 (provavelmente para adequar databases diferentes). Mas o Guitarset tem sample rate 44100, então isso pode ser aproveitado.

SAVED_MODELS_BASE_PATH="saved_models"

# Total de semitons das anotações.
NUM_ANNOTATION_SEMITONES = 88
MINIMUM_ANNOTATION_FREQUENCY = 27.5 # em Hz.


CQT_HOP_LENGTH = 256 # Se for usado 44100, esse deve ser dobrado também.

# Harmônicos adotados na HCQT.
HARMONICS_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7]
MAX_HARMONIC = max(HARMONICS_LIST)

# Bins das anotações de pitch (contours).
CONTOURS_BINS_PER_SEMITONE = 3
CONTOURS_BINS_PER_OCTAVE = CONTOURS_BINS_PER_SEMITONE * 12
CONTOURS_TOTAL_BINS = NUM_ANNOTATION_SEMITONES * CONTOURS_BINS_PER_SEMITONE

# Bins das anotações de notas e onsets.
NOTES_BINS_PER_SEMITONE = 1 
NOTES_BINS_PER_OCTAVE = NOTES_BINS_PER_SEMITONE * 12
NOTES_TOTAL_BINS = NUM_ANNOTATION_SEMITONES * NOTES_BINS_PER_SEMITONE

# Número de semitons e bins calculados para a CQT. Desejaria-se obter todos os harmônicos para a maior frequência,
# mas existe uma limitação dada pela frequência de Nyquist.
DESIRED_CQT_SEMITONES = NUM_ANNOTATION_SEMITONES + int(ceil(12.0 * log2(MAX_HARMONIC)) + NUM_ANNOTATION_SEMITONES) 
NYQUIST_MAXIMUM_CQT_SEMITONES = int(floor(12.0 * log2(0.5 * AUDIO_SAMPLE_RATE / MINIMUM_ANNOTATION_FREQUENCY)))
NUM_CQT_SEMITONES = min(DESIRED_CQT_SEMITONES, NYQUIST_MAXIMUM_CQT_SEMITONES)
CQT_TOTAL_BINS = NUM_CQT_SEMITONES * CONTOURS_BINS_PER_SEMITONE


AUDIO_SEGMENT_LEN_SECS = 2.0 # Segmementos em segundos
AUDIO_SEGMENT_LEN_FRAMES = int(AUDIO_SEGMENT_LEN_SECS * AUDIO_SAMPLE_RATE // CQT_HOP_LENGTH) # Também pode ser obtido com librosa.time_to_frames
AUDIO_SEGMENT_LEN_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_SEGMENT_LEN_SECS - CQT_HOP_LENGTH)


NUM_TRACKS_PER_RECORD_FILE = 8



# Parâmetros de treinamento.
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SHUFFLE_BUFFER = 100
DEFAULT_BATCH = 32
DEFAULT_EPOCHS = 100
DEFAULT_LABEL_SMOOTHING = 0.2
DEFAULT_ONSET_POSITIVE_WEIGHT = 0.95