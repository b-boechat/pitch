import os # Temporário, isso vai estar no main.py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import json
from model import get_compiled_model
from deserialize_guitarset import prepare_dataset

def train(
        learning_rate, label_smoothing, buffer_size, batch_size, onset_positive_weight, epochs, verbose, 
        data_base_path, output_path, plot_history):

    model = get_compiled_model(learning_rate = learning_rate, label_smoothing = label_smoothing, 
                               onset_positive_weight = onset_positive_weight, plot_summary = True)

    train_path_list = glob.glob(f"{data_base_path}/training/" + "*.tfrecord")
    dataset = prepare_dataset(train_path_list, buffer_size = buffer_size, batch_size = batch_size)

    val_path_list = glob.glob(f"{data_base_path}/test/" + "*.tfrecord")
    val_dataset = prepare_dataset(val_path_list, buffer_size = buffer_size, batch_size = batch_size)

    history = model.fit(
        dataset,
        epochs = epochs,
        verbose = verbose,
        validation_data = val_dataset
    )

    print("Training dataset:")
    model.evaluate(dataset, verbose = verbose)
    print("Validation dataset:")
    model.evaluate(val_dataset, verbose = verbose)

    if output_path is not None:
        model.save_weights(f"{output_path}.h5", overwrite = True, save_format = "h5")
        if plot_history:
            json.dump(history.history, open(f"{output_path}.json", 'w'))


if __name__ == "__main__":
    with tf.device('/GPU:1'):
        print("Model: lr = 1e-3")
        train(learning_rate = 0.001, label_smoothing = 0.2, onset_positive_weight = 0.95, buffer_size = 100, batch_size = 32, epochs = 200, output_path = "saved_models/test.h5", verbose=1, data_base_path="swgm")

