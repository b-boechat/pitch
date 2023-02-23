import os # Temporário, isso vai estar no main.py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import glob
from model import get_compiled_model
from deserialize_guitarset import prepare_dataset

def train_prelim(learning_rate, label_smoothing, buffer_size, batch_size, onset_positive_weight, epochs, verbose = 0, save_path=None):

    model = get_compiled_model(learning_rate = learning_rate, label_smoothing = label_smoothing, 
                               onset_positive_weight = onset_positive_weight, plot_summary = True)

    data_path_list = glob.glob(r"guitarset_processed/training/" + r"/*.tfrecord")
    dataset = prepare_dataset(data_path_list, buffer_size = buffer_size, batch_size = batch_size)

    val_path_list = glob.glob(r"guitarset_processed/test/" + r"/*.tfrecord")
    val_dataset = prepare_dataset(val_path_list, buffer_size = buffer_size, batch_size = batch_size)

    model.fit(
        dataset,
        epochs = epochs,
        verbose = verbose,
        validation_data = dataset
    )

    print("Training dataset:")
    model.evaluate(dataset, verbose = verbose)
    print("Validation dataset:")
    model.evaluate(val_dataset, verbose = verbose)

    if save_path is not None:
        model.save_weights(save_path, overwrite = True, save_format = "h5")


if __name__ == "__main__":
    with tf.device('/GPU:1'):
        train_prelim(verbose = 1, learning_rate = 0.0001, label_smoothing = 0.2, onset_positive_weight = 0.95, buffer_size = 100, batch_size = 16, epochs = 20, save_path="saved_models/testchanges.h5")
        # print("Model: lr = 1e-2")
        # train_prelim(verbose = 1, learning_rate = 0.01, label_smoothing = 0.2, onset_positive_weight = 0.95, buffer_size = 100, batch_size = 16, epochs = 100, save_path="saved_models/trained_lr_e2.h5")
        # print("Model: lr = 1e-3")
        # train_prelim(verbose = 1, learning_rate = 0.001, label_smoothing = 0.2, onset_positive_weight = 0.95, buffer_size = 100, batch_size = 16, epochs = 100, save_path = "saved_models/trained_lr_e3.h5")
        # print("Model: lr = 1e-4")
        # train_prelim(verbose = 1, learning_rate = 0.0001, label_smoothing = 0.2, onset_positive_weight = 0.95, buffer_size = 100, batch_size = 32, epochs = 100, save_path = "saved_models/trained_lr_e4.h5")
        # print("Model: lr = 1e-5")
        # train_prelim(verbose = 1, learning_rate = 0.00001, label_smoothing = 0.2, onset_positive_weight = 0.95, buffer_size = 100, batch_size = 32, epochs = 100, save_path = "saved_models/trained_lr_e5.h5")