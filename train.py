import os # Temporário, isso vai estar no main.py
import tensorflow as tf
import glob
from model import define_model, get_loss_dictionary
from deserialize_guitarset import prepare_dataset

def train_prelim(lr, label_smoothing, buffer_size, batch_size, onset_positive_weight, epochs, verbose = 0, save_path=None):

    model = define_model()
    model.compile(loss=get_loss_dictionary(label_smoothing, onset_positive_weight), 
                optimizer=tf.keras.optimizers.Adam(learning_rate = lr))

    data_path_list = glob.glob(r"guitarset_processed/training/" + r"/*.tfrecord")
    dataset = prepare_dataset(data_path_list, buffer_size=buffer_size, batch_size=batch_size)

    val_path_list = glob.glob(r"guitarset_processed/test/" + r"/*.tfrecord")
    val_dataset = prepare_dataset(val_path_list, buffer_size=buffer_size, batch_size=batch_size)

    if verbose == 2:
        #model.summary(show_trainable=True)
        model.summary()

    model.fit(
        dataset,
        epochs = epochs,
        verbose = verbose,
        validation_data = dataset
    )

    model.evaluate(val_dataset, verbose = verbose)

    if save_path is not None:
        model.save_weights(save_path, overwrite=True, save_format="h5")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with tf.device('/GPU:1'):
        #train_prelim(verbose = 1)
        #train_prelim(verbose = 1, lr=0.001, label_smoothing=0.2, onset_positive_weight=0.95, buffer_size=100, batch_size=16, epochs=100, save_path="saved_models/trained.h5")
        print("First model.")
        train_prelim(verbose = 1, lr=0.0001, label_smoothing=0.2, onset_positive_weight=0.95, buffer_size=100, batch_size=32, epochs=100, save_path="saved_models/trained_lr_e4.h5")
        print("Second model.")
        train_prelim(verbose = 1, lr=0.00001, label_smoothing=0.2, onset_positive_weight=0.95, buffer_size=100, batch_size=32, epochs=100, save_path="saved_models/trained_lr_e5.h5")
        print("Third model.")
        train_prelim(verbose = 1, lr=0.00001, label_smoothing=0.2, onset_positive_weight=0.95, buffer_size=100, batch_size=32, epochs=100, save_path="saved_models/trained_lr_e3.h5")