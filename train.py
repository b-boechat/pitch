import tensorflow as tf
import glob
from model import define_model, get_loss_dictionary
from deserialize_guitarset import prepare_dataset

def train_prelim(lr = 0.001, label_smoothing = 0.2, buffer_size = 10, batch_size = 4, onset_positive_weight = 0.5, verbose = 0, epochs = 4, save_path=None):

    model = define_model()
    model.compile(loss=get_loss_dictionary(label_smoothing, onset_positive_weight), 
                optimizer=tf.keras.optimizers.Adam(learning_rate = lr))

    data_path_list = glob.glob(r"guitarset_processed/training/" + r"/*.tfrecord")
    dataset = prepare_dataset(data_path_list, buffer_size=buffer_size, batch_size=batch_size)

    if verbose == 2:
        #model.summary(show_trainable=True)
        model.summary()

    model.fit(
        dataset,
        epochs = epochs,
        verbose = verbose,
        validation_data = dataset
    )

    if save_path is not None:
        model.save_weights(save_path, overwrite=True)


if __name__ == "__main__":
    with tf.device('/GPU:1'):
        #train_prelim(verbose = 1)
        train_prelim(verbose = 1, buffer_size=100, batch_size=16, epochs=100, save_path="saved_models/backup")