import json
from utils import create_unique_folder, get_folder_path
from deserialize_guitarset import prepare_dataset_cv, get_cv_filenames, get_split_filenames
from train import _fit, _save_model
from model import restore_model_from_weights
from evaluate_model import _get_eval_output_json_path, mir_evaluate_model_on_files
from definitions import DEFAULT_VAL_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME

def train_cv(
        learning_rate, label_smoothing, buffer_size, batch_size, onset_positive_weight, epochs, verbose, 
        data_base_dir, output_cv_folder_id, save_history, num_cv_groups, output_base_path, cv_split_name, model_index_to_resume
    ):

    if model_index_to_resume is None:
        output_cv_folder_path = create_unique_folder(output_base_path, output_cv_folder_id, verbose=verbose)
        cv_meta_dict = {
            "learning_rate": learning_rate,
            "label_smoothing": label_smoothing,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "onset_positive_weight": onset_positive_weight,
            "epochs": epochs,
            "data_base_dir": data_base_dir,
            "num_cv_groups": num_cv_groups,
            "groups": [get_cv_filenames(data_base_dir, cv_split_name, num_cv_groups, i, all_except_group=False) for i in range(num_cv_groups)]
        }
        json.dump(cv_meta_dict, open(f"{output_cv_folder_path}/{output_cv_folder_id}_cv_meta.json", 'w'))
    else:
        output_cv_folder_path = get_folder_path(output_base_path, output_cv_folder_id)

    for i in range(num_cv_groups):
        if model_index_to_resume is not None and i < model_index_to_resume:
            continue
        train_dataset = prepare_dataset_cv(
            data_base_dir=data_base_dir, 
            cv_split_name="cv",  
            num_cv_groups=num_cv_groups,
            cv_index=i,
            is_train_dataset=True,
            buffer_size = buffer_size, 
            batch_size = batch_size
        )

        val_dataset = prepare_dataset_cv(
            data_base_dir=data_base_dir, 
            cv_split_name="cv",  
            num_cv_groups=num_cv_groups,
            cv_index=i,
            is_train_dataset=False,
            buffer_size = buffer_size, 
            batch_size = batch_size
        )

        model, history = _fit(train_dataset, val_dataset, learning_rate, label_smoothing, onset_positive_weight, epochs, batch_size, verbose)

        if not save_history:
            history = None

        _save_model(
            model=model, 
            output_base_path=output_cv_folder_path, 
            output_folder_id=f"{output_cv_folder_id}_group_{i}",
            meta_dict={
                "learning_rate": learning_rate,
                "label_smoothing": label_smoothing,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "onset_positive_weight": onset_positive_weight,
                "epochs": epochs,
                "data_base_dir": data_base_dir,
                "group": i
            },
            history=history, 
            verbose=verbose
        )

def evaluate_model_cv(cv_folder_id, cv_split_name, split, onset_threshold_list, frame_threshold_list, saved_models_base_path, verbose):
    assert split in (DEFAULT_VAL_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME) # TODO For now, it's assumed that these are the splits names used (or implied, in the cal of "val" in a "cv" split).
    
    cv_base_path = f"{saved_models_base_path}/{cv_folder_id}"
    cv_meta = json.load(open(f"{cv_base_path}/{cv_folder_id}_cv_meta.json", 'r'))

    num_cv_groups = cv_meta["num_cv_groups"]
    data_base_dir = cv_meta["data_base_dir"]
    
    for i in range(num_cv_groups):
        
        model_id = f"{cv_folder_id}_group_{i}"
        model = restore_model_from_weights(
                                saved_model_path = f"{cv_base_path}/{model_id}/{model_id}.h5",
                                label_smoothing = cv_meta["label_smoothing"],
                                onset_positive_weight = cv_meta["onset_positive_weight"] 
                            )
        
        if verbose >= 1:
            print(f"Evaluating {model_id}.")
        
        if split == DEFAULT_VAL_SPLIT_NAME:
            # Validation split refers to the i-th group of the cv split.
            ds_files = get_cv_filenames(
                                    data_base_dir = data_base_dir, 
                                    cv_split_name = cv_split_name, 
                                    num_cv_groups = num_cv_groups, 
                                    cv_index = i, 
                                    all_except_group = False
                                )
        else: # Used for test split.
            ds_files = get_split_filenames(data_base_dir, split_name=split)
        
        for onset_t in onset_threshold_list:
            for frame_t in frame_threshold_list:
                metrics_meta = [{"split_name": split, "onset_threshold": onset_t, "frame_threshold": frame_t}]
                if verbose >= 1:
                    print(metrics_meta)
                
                metrics_list = mir_evaluate_model_on_files(
                    model = model,
                    file_path = ds_files,
                    onset_threshold = onset_t,
                    frame_threshold = frame_t,
                    verbosity = verbose
                )
                eval_output_path = _get_eval_output_json_path(
                    base_path = cv_base_path,
                    model_id = model_id,
                    split_name = split,
                    onset_threshold = onset_t,
                    frame_threshold = frame_t
                )
                
                if verbose >= 1:
                    print(f"Writing to {eval_output_path}", end="\n\n")
                json.dump(metrics_meta + metrics_list, open(eval_output_path, 'w'))

