import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.backend import int_shape

from definitions import AUDIO_SEGMENT_LEN_FRAMES, CONTOURS_BINS_PER_SEMITONE, CONTOURS_TOTAL_BINS, CONTOURS_TOTAL_BINS, CQT_TOTAL_BINS, HARMONICS_LIST


#def _kernel_initializer(dtype) -> tf.keras.initializers.VarianceScaling:
#    return tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None)
#def _kernel_constraint(dtype) -> tf.keras.constraints.UnitNorm:
#    return tf.keras.constraints.UnitNorm(axis=[0, 1, 2])

def _initializer() -> tf.keras.initializers.VarianceScaling:
    return tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None)

class FlattenFreqCh(tf.keras.layers.Layer):
    """Layer to flatten the frequency channel and make each channel
    part of the frequency dimension.

    Input shape: (batch, time, freq, ch)
    Output shape: (batch, time, freq*ch)
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        shapes = int_shape(x)
        return tf.keras.layers.Reshape([shapes[1], shapes[2] * shapes[3]])(x)  # ignore batch size


def log_base_b(x: tf.Tensor, base: int) -> tf.Tensor:
    """
    Compute log_b(x)
    Args:
        x : input
        base : log base. E.g. for log10 base=10
    Returns:
        log_base(x)
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


class HarmonicStacking(tf.keras.layers.Layer):
    """Harmonic stacking layer

    Input shape: (n_batch, n_times, n_freqs, 1)
    Output shape: (n_batch, n_times, n_output_freqs, len(harmonics))

    n_freqs should be much larger than n_output_freqs so that information from the upper
    harmonics is captured.

    Attributes:
        bins_per_semitone: The number of bins per semitone of the input CQT
        harmonics: List of harmonics to use. Should be positive numbers.
        shifts: A list containing the number of bins to shift in frequency for each harmonic
        n_output_freqs: The number of frequency bins in each harmonic layer.
    """

    def __init__(self, bins_per_semitone: int, harmonics, n_output_freqs: int, name: str = "harmonic_stacking"):
        """Downsample frequency by stride, upsample channels by 4."""
        super().__init__(trainable=False, name=name)
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.shifts = [
            int(tf.math.round(12.0 * self.bins_per_semitone * log_base_b(float(h), 2))) for h in self.harmonics
        ]
        self.n_output_freqs = n_output_freqs

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "bins_per_semitone": self.bins_per_semitone,
                "harmonics": self.harmonics,
                "n_output_freqs": self.n_output_freqs,
                "name": self.name,
            }
        )
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # (n_batch, n_times, n_freqs, 1)
        tf.debugging.assert_equal(tf.shape(x).shape, 4)
        channels = []
        for shift in self.shifts:
            if shift == 0:
                padded = x
            elif shift > 0:
                paddings = tf.constant([[0, 0], [0, 0], [0, shift], [0, 0]])
                padded = tf.pad(x[:, :, shift:, :], paddings)
            elif shift < 0:
                paddings = tf.constant([[0, 0], [0, 0], [-shift, 0], [0, 0]])
                padded = tf.pad(x[:, :, :shift, :], paddings)
            else:
                raise ValueError

            channels.append(padded)
        x = tf.concat(channels, axis=-1)
        x = x[:, :, : self.n_output_freqs, :]  # return only the first n_output_freqs frequency channels
        return x


def define_model():

    # Define input shape. At the moment, considering CQT as input.
    inputs = keras.Input(shape=(AUDIO_SEGMENT_LEN_FRAMES, CQT_TOTAL_BINS, 1))

    # Batch norm CQT and create HCQT (non-trainable layer)
    x = layers.BatchNormalization()(inputs)
    x_stack = HarmonicStacking(
            bins_per_semitone=CONTOURS_BINS_PER_SEMITONE,
            harmonics=HARMONICS_LIST,
            n_output_freqs=CONTOURS_TOTAL_BINS,
    )(x)

    # =========== Contour layers ============

    # First layer. 16 Conv2D 5 x 5 + Batch norm + ReLu
    x = layers.Conv2D(
        filters=16,
        kernel_size=(5,5),
        padding='same' # Pad with zeros so output has the same size as input.
        #kernel_initializer=_initializer,
        #kernel_constraint=_kernel_constraint
    )(x_stack)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second layer. 8 Conv2D 3 x 39 + Batch norm + ReLu
    x = layers.Conv2D(
        filters=8,
        kernel_size=(3,39),
        padding='same' # Pad with zeros so output has the same size as input.
        #kernel_initializer=_kernel_initializer, # Initialize weights.
        #kernel_constraint=_kernel_constraint # Constraint function applied to the kernel.
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Third and final layer. 1 Conv2D 5 x 5 + Sigmoid + Flatten frequencies
    x = layers.Conv2D(
        filters=1,
        kernel_size=(5,5),
        padding='same',
        activation='sigmoid'
        #kernel_initializer=_kernel_initializer,
        #kernel_constraint=_kernel_constraint
    )(x)

    x_contours = FlattenFreqCh()(x)

    # =========== Note layers ============

    x = tf.expand_dims(x_contours, -1)

    # First layer. 32 Conv2D 7 x 7, stride 1 x 3 + ReLu
    x = layers.Conv2D(
        filters=32,
        kernel_size=(7,7),
        strides=(1, 3),
        padding='same' # Pad with zeros so output has the same time size as input, and 1/3 frequency size.
        #kernel_initializer=_kernel_initializer,
        #kernel_constraint=_kernel_constraint
    )(x)
    x = layers.ReLU()(x)

    # Second layer. 1 Conv2D 7 x 3 + Sigmoid + Flatten frequencies
    x_notes_stacked = layers.Conv2D(
        filters=1,
        kernel_size=(7,3),
        padding='same', # Pad with zeros so output has the same size as input.
        activation='sigmoid'
        #kernel_initializer=_kernel_initializer,
        #kernel_constraint=_kernel_constraint
    )(x)

    x_notes = FlattenFreqCh()(x_notes_stacked)

    # =========== Onset layers ============

    # First layer. 32 Conv2D 7 x 7, stride 1 x 3, HCQT input + Batch norm + ReLu
    x = layers.Conv2D(
        filters=32,
        kernel_size=(7,7),
        strides=(1, 3),
        padding='same' # Pad with zeros so output has the same size as input.
        #kernel_initializer=_kernel_initializer,
        #kernel_constraint=_kernel_constraint
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
        activation='sigmoid'
        #kernel_initializer=_kernel_initializer,
        #kernel_constraint=_kernel_constraint
    )(x)

    x_onsets = FlattenFreqCh()(x)

    outputs = {"contours": x_contours, "notes": x_notes, "onsets": x_onsets}
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()


define_model()
