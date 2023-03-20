from note_creation import model_output_to_notes
import glob
from model import restore_model_from_weights
from definitions import *
from deserialize_guitarset import *

def test_note_creation(model_path, data_path_list):
    model = restore_model_from_weights(model_path, learning_rate=DEFAULT_LEARNING_RATE, label_smoothing=DEFAULT_LABEL_SMOOTHING, onset_positive_weight=DEFAULT_ONSET_POSITIVE_WEIGHT)
    ds = fetch_dataset(data_path_list)

    for example in ds.take(10):
        id, X_spec, X_contours, X_notes, X_onsets = example
        example_dict = {'notes': X_notes, 'onsets': X_onsets, 'contours': X_contours}
        midi, events = model_output_to_notes(example_dict, onset_thresh=0.5, frame_thresh=0.5)
        #print(midi)
        print([event[2] for event in events])
        #print(events, "\n\n")
        pred_dict = output_batch_to_single(
                        model.predict_on_batch(
                            np.expand_dims(X_spec, axis=0)))
        pred_dict = {'notes': pred_dict["X_notes"], 'onsets': pred_dict["X_onsets"], "contours": pred_dict["X_contours"]}
        midi_pred, events_pred = model_output_to_notes(pred_dict, onset_thresh=0.8, frame_thresh=0.8)
        #print(midi_pred)
        #print(events_pred)
        print([event[2] for event in events_pred], end="\n\n\n")
        

if __name__ == "__main__":
    test_note_creation("saved_models/trained_lr_e3.h5", glob.glob(r"guitarset_processed/test/" + r"/*.tfrecord"))
