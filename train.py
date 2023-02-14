import tensorflow as tf
from model_profiler import model_profiler
from sys import getsizeof
from model import define_model, get_loss_dictionary
from deserialize_guitarset import formatted_dataset_for_model

def train_prelim(lr = 0.001, label_smoothing = 0.2, batch_size = 3, onset_positive_weight = 0.5, verbose = 0, profile=False):

    model = define_model()
    model.compile(loss=get_loss_dictionary(label_smoothing, onset_positive_weight), 
                optimizer=tf.keras.optimizers.Adam(learning_rate = lr)
                )

    dataset = formatted_dataset_for_model(["guitarset_processed/training/split_" + number + ".tfrecord" for number in ["000"]])

    if profile:
        print(f"Size (1): {getsizeof(dataset)}")

    dataset = dataset.shuffle(buffer_size=10).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    if profile:
        print(f"Size (2): {getsizeof(dataset)}")

    if verbose:
        model.summary()
        print("\n\n")

    if profile:
        print(model_profiler(model, batch_size))
        return

    model.fit(
        dataset,
        epochs=4,
        verbose = verbose,
        validation_data = dataset
    )




if __name__ == "__main__":
    #train_prelim(verbose = 2, profile=True)
    train_prelim(verbose = 2)