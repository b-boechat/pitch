import pandas as pd
import json
from glob import glob
from definitions import SAVED_MODELS_BASE_PATH

def read_metrics(model_id, split_name="val", keys="all"):
    assert split_name in ("train", "val", "test")

    model_folder = f"{SAVED_MODELS_BASE_PATH}/{model_id}"
    best_f_measure = -1.0
    best_onset_threshold, best_frame_threshold = -1., -1.

    for path in sorted(glob(f"{model_folder}/{model_id}_eval_{split_name}*.json")): # TODO move this glob call to a different function.
        dump = json.load(open(path, 'r'))
        meta = dump[0]
        records = dump[1:]
        df = pd.DataFrame.from_records(records)
        print(f"onset thresh: {meta['onset_threshold']}, frame thresh: {meta['frame_threshold']}")

        if keys == "all":
            keys = [key for key in df.keys() if key != "id"]


        for key in keys:
            key_mean = df[key].mean()
            key_std = df[key].std()
            if key == "F-measure_no_offset" and key_mean > best_f_measure:
                best_f_measure = key_mean
                best_onset_threshold, best_frame_threshold = meta['onset_threshold'], meta['frame_threshold']
            print(f"{key}, mean: {key_mean}, std: {key_std}")
        print(end="\n\n")
    print(f"Best thresholds: ({best_onset_threshold}, {best_frame_threshold}). F_measure: {100*best_f_measure}%")