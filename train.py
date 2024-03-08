import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json
from uuid import uuid4
from model import get_compiled_model
from deserialize_guitarset import prepare_dataset_split
from utils import create_unique_folder

def train(
        learning_rate, label_smoothing, buffer_size, batch_size, onset_positive_weight, epochs, verbose, 
        data_base_dir, output_folder_id, save_history, output_base_path
    ):
    
    train_dataset = prepare_dataset_split(data_base_dir, "train", buffer_size = buffer_size, batch_size = batch_size)
    val_dataset = prepare_dataset_split(data_base_dir, "val", buffer_size = buffer_size, batch_size = batch_size)

    model, history = _fit(train_dataset, val_dataset, learning_rate, label_smoothing, onset_positive_weight, epochs, batch_size, verbose)

    if output_folder_id is not None:
        output_folder_path = create_unique_folder(output_base_path, output_folder_id, verbose)
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

def _fit(train_dataset, val_dataset, learning_rate, label_smoothing, onset_positive_weight, epochs, batch_size, verbose):

    model = get_compiled_model(learning_rate = learning_rate, label_smoothing = label_smoothing, onset_positive_weight = onset_positive_weight, plot_summary = verbose)

    history = model.fit(
        train_dataset,
        epochs = epochs,
        verbose = verbose,
        validation_data = val_dataset.take(batch_size//4),
        validation_steps = 5
    )

    if verbose >= 1:
        print("Training dataset:")
        model.evaluate(train_dataset, verbose = verbose)
        print("Validation dataset:")
        model.evaluate(val_dataset, verbose = verbose)

    return model, history    