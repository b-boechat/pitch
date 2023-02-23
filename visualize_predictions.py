import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import glob
from model import restore_model_from_weights
from definitions import DEFAULT_LEARNING_RATE, DEFAULT_LABEL_SMOOTHING, DEFAULT_ONSET_POSITIVE_WEIGHT
from deserialize_guitarset import fetch_dataset, zip_for_model, zipped_single_to_batch, output_batch_to_single

def visualize_predictions(saved_model_path, data_path_list):
    model = restore_model_from_weights(saved_model_path, learning_rate = DEFAULT_LEARNING_RATE, label_smoothing = DEFAULT_LABEL_SMOOTHING,  
                                       onset_positive_weight = DEFAULT_ONSET_POSITIVE_WEIGHT)

    dataset = fetch_dataset(data_path_list)
    for example in dataset.take(1):
        #print(example, end="\n\n\n")
        pred = output_batch_to_single(
                model.predict_on_batch(
                    zipped_single_to_batch( 
                        zip_for_model(*example))))


if __name__ == "__main__":
    visualize_predictions("saved_models/trained_lr_e3.h5", glob.glob(r"guitarset_processed/training/" + r"/*.tfrecord"))
    
                        
    