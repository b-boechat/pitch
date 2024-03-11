import json
from utils import create_unique_folder, get_folder_path
from deserialize_guitarset import prepare_dataset_cv, get_cv_filenames
from train import _fit, _save_model

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

#def evaluate_model(cv_folder_id, split_name, onset_threshold, frame_threshold, base_path, verbosity):