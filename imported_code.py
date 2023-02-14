import tensorflow as tf
from keras.backend import int_shape


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


def weighted_transcription_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, label_smoothing: float, positive_weight: float = 0.5
) -> tf.Tensor:
    """The transcription loss where the positive and negative true labels are balanced by a weighting factor.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        The weighted transcription loss.
    """
    negative_mask = tf.equal(y_true, 0)
    nonnegative_mask = tf.logical_not(negative_mask)
    bce_negative = tf.keras.losses.binary_crossentropy(
        tf.boolean_mask(y_true, negative_mask),
        tf.boolean_mask(y_pred, negative_mask),
        label_smoothing=label_smoothing,
    )
    bce_nonnegative = tf.keras.losses.binary_crossentropy(
        tf.boolean_mask(y_true, nonnegative_mask),
        tf.boolean_mask(y_pred, nonnegative_mask),
        label_smoothing=label_smoothing,
    )
    return ((1 - positive_weight) * bce_negative) + (positive_weight * bce_nonnegative)


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