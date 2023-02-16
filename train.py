import tensorflow as tf
from model_profiler import model_profiler
from sys import getsizeof
from model import define_model, get_loss_dictionary
from deserialize_guitarset import prepare_dataset

def train_prelim(lr = 0.001, label_smoothing = 0.2, buffer_size = 10, batch_size = 4, onset_positive_weight = 0.5, verbose = 0, epochs = 4, profile=False):

    model = define_model()
    model.compile(loss=get_loss_dictionary(label_smoothing, onset_positive_weight), 
                optimizer=tf.keras.optimizers.Adam(learning_rate = lr))

    dataset = prepare_dataset(["guitarset_processed/training/split_" + number + ".tfrecord" for number in ["000", "001", "002"]], 
                    buffer_size=buffer_size, batch_size=batch_size)

    if verbose == 2:
        model.summary(show_trainable=True)

    if profile:
        print(model_profiler(model, batch_size))
        return

    model.fit(
        dataset,
        epochs = epochs,
        verbose = verbose,
        validation_data = dataset
    )

if __name__ == "__main__":
    #train_prelim(verbose = 2, profile=True)
    train_prelim(verbose = 2)