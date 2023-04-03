import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import json
from uuid import uuid4
from model import get_compiled_model
from deserialize_guitarset import prepare_dataset
from definitions import PROCESSED_DATASETS_BASE_PATH, SAVED_MODELS_BASE_PATH

def train(
        learning_rate, label_smoothing, buffer_size, batch_size, onset_positive_weight, epochs, verbose, 
        data_base_dir, output_folder_id, save_history):

    model = get_compiled_model(learning_rate = learning_rate, label_smoothing = label_smoothing, 
                               onset_positive_weight = onset_positive_weight, plot_summary = True)

    train_path_list = glob.glob(f"{PROCESSED_DATASETS_BASE_PATH}/{data_base_dir}/train/*.tfrecord")
    dataset = prepare_dataset(train_path_list, buffer_size = buffer_size, batch_size = batch_size)

    val_path_list = glob.glob(f"{PROCESSED_DATASETS_BASE_PATH}/{data_base_dir}/val/*.tfrecord")
    val_dataset = prepare_dataset(val_path_list, buffer_size = buffer_size, batch_size = batch_size)

    history = model.fit(
        dataset,
        epochs = epochs,
        verbose = verbose,
        validation_data = val_dataset,
        validation_steps = 5
    )

    print("Training dataset:")
    model.evaluate(dataset, verbose = verbose)
    print("Validation dataset:")
    model.evaluate(val_dataset, verbose = verbose)

    if output_folder_id is not None:
        output_folder_path = f"{SAVED_MODELS_BASE_PATH}/{output_folder_id}"
        if os.path.isdir(output_folder_path):
            print(f"Output folder '{output_folder_id}' exists.")
            output_folder_id = f"{output_folder_id}-{uuid4().hex}"
            output_folder_path = f"{SAVED_MODELS_BASE_PATH}/{output_folder_id}"
        os.mkdir(output_folder_path)
        print(f"Writing on '{output_folder_id}'.")
        model.save_weights(f"{output_folder_path}/{output_folder_id}.h5", overwrite = True, save_format = "h5")
        json.dump({
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "onset_positive_weight": onset_positive_weight,
            "epochs": epochs,
            "data_base_dir": data_base_dir
        }, open(f"{output_folder_path}/{output_folder_id}_meta.json", 'w'))
        if save_history:
            json.dump(history.history, open(f"{output_folder_path}/{output_folder_id}_history.json", 'w'))

if __name__ == "__main__":
    with tf.device('/GPU:1'):
        print("Model: lr = 1e-3")
        train(learning_rate = 0.001, label_smoothing = 0.2, onset_positive_weight = 0.95, buffer_size = 100, batch_size = 32, epochs = 50, output_folder_id = "saved_models/test", verbose=1, data_base_dir="swgm")

