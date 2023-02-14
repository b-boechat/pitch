import tensorflow as tf
from keras import layers
from keras.backend import int_shape

from definitions import AUDIO_SEGMENT_LEN_FRAMES, CONTOURS_BINS_PER_SEMITONE, CONTOURS_TOTAL_BINS, CONTOURS_TOTAL_BINS, CQT_TOTAL_BINS, HARMONICS_LIST
from imported_code import FlattenFreqCh, HarmonicStacking, weighted_transcription_loss

def define_model(plot_summary=False):

    # Define input shape. At the moment, considering CQT as input.
    inputs = tf.keras.Input(shape=(AUDIO_SEGMENT_LEN_FRAMES, CQT_TOTAL_BINS))

    # Batch norm CQT and create HCQT (non-trainable layer)
    x = tf.expand_dims(inputs, -1)
    x = layers.BatchNormalization()(x)
    x_stack = HarmonicStacking(
            bins_per_semitone=CONTOURS_BINS_PER_SEMITONE,
            harmonics=HARMONICS_LIST,
            n_output_freqs=CONTOURS_TOTAL_BINS,
    )(x)

    # =========== Contour layers ============

    # First layer. 16 Conv2D 5 x 5 + Batch norm + ReLu
    x = layers.Conv2D(
        #filters=16,
        filters=1,
        kernel_size=(5,5),
        padding='same', # Pad with zeros so output has the same size as input.
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2])
    )(x_stack)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second layer. 8 Conv2D 3 x 39 + Batch norm + ReLu
    x = layers.Conv2D(
        #filters=8,
        filters=1,
        kernel_size=(3,39),
        padding='same', # Pad with zeros so output has the same size as input.
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2])
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Third and final layer. 1 Conv2D 5 x 5 + Sigmoid + Flatten frequencies
    x = layers.Conv2D(
        filters=1,
        kernel_size=(5,5),
        padding='same',
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2])
    )(x)

    x_contours = FlattenFreqCh()(x)

    # =========== Note layers ============

    x = tf.expand_dims(x_contours, -1)

    # First layer. 32 Conv2D 7 x 7, stride 1 x 3 + ReLu
    x = layers.Conv2D(
        #filters=32,
        filters=1,
        kernel_size=(7,7),
        strides=(1, 3),
        padding='same', # Pad with zeros so output has the same time size as input, and 1/3 frequency size.
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2])
    )(x)
    x = layers.ReLU()(x)

    # Second layer. 1 Conv2D 7 x 3 + Sigmoid + Flatten frequencies
    x_notes_stacked = layers.Conv2D(
        filters=1,
        kernel_size=(7,3),
        padding='same', # Pad with zeros so output has the same size as input.
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2])
    )(x)

    x_notes = FlattenFreqCh()(x_notes_stacked)

    # =========== Onset layers ============

    # First layer. 32 Conv2D 7 x 7, stride 1 x 3, HCQT input + Batch norm + ReLu
    x = layers.Conv2D(
        #filters=32,
        filters=1,
        kernel_size=(7,7),
        strides=(1, 3),
        padding='same', # Pad with zeros so output has the same size as input.
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2])
    )(x_stack)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Concatenate with x_notes before flattening.
    x = layers.Concatenate(axis=3)([x_notes_stacked, x])

    # Second layer. 1 Conv2D 3 x 3 + Sigmoid + Flatten frequencies
    x = layers.Conv2D(
        filters=1,
        kernel_size=(3,3),
        padding='same', # Pad with zeros so output has the same size as input.
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None),
        kernel_constraint=tf.keras.constraints.UnitNorm(axis=[0, 1, 2])
    )(x)

    x_onsets = FlattenFreqCh()(x)

    outputs = {"X_contours": x_contours, "X_notes": x_notes, "X_onsets": x_onsets}
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if plot_summary:
        model.summary()
    return model


def non_weighted_transcription_loss(y_true, y_pred, label_smoothing):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)

def get_loss_dictionary(label_smoothing, onset_positive_weight):
    loss_contours = lambda y_true, y_pred: non_weighted_transcription_loss(y_true, y_pred, label_smoothing)
    loss_notes = loss_contours
    loss_onsets = lambda y_true, y_pred: weighted_transcription_loss(y_true, y_pred, label_smoothing, onset_positive_weight)
    return {
        "X_contours": loss_contours,
        "X_notes": loss_notes,
        "X_onsets": loss_onsets
    }

if __name__ == "__main__":
    define_model(plot_summary=True)
