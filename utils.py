import os
from uuid import uuid4

def create_unique_folder(folder_base_path, folder_id, verbose=1):
    folder_path = get_folder_path(folder_base_path, folder_id)
    if os.path.isdir(folder_path):
        if verbose >= 1:
            print(f"Output folder '{folder_id}' exists.")
        folder_id = f"{folder_id}-{uuid4().hex}"
        if verbose >= 1:
            print(f"Writing to '{folder_id}' instead.")
        folder_path = f"{folder_base_path}/{folder_id}"
    os.mkdir(folder_path)
    return folder_path

def get_folder_path(folder_base_path, folder_id):
    return f"{folder_base_path}/{folder_id}"