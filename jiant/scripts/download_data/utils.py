import datasets
import os
import tarfile
import urllib
import zipfile

import jiant.utils.python.io as py_io
from jiant.utils.python.datastructures import replace_key


def convert_hf_dataset_to_examples(
    path, name=None, version=None, field_map=None, label_map=None, phase_map=None, phase_list=None
):
    """Helper function for reading from datasets.load_dataset and converting to examples

    Args:
        path: path argument (from datasets.load_dataset)
        name: name argument (from datasets.load_dataset)
        version: version argument (from datasets.load_dataset)
        field_map: dictionary for renaming fields, non-exhaustive
        label_map: dictionary for replacing labels, non-exhaustive
        phase_map: dictionary for replacing phase names, non-exhaustive
        phase_list: phases to keep (after phase_map)

    Returns:
        Dict[phase] -> list[examples]
    """
    dataset = datasets.load_dataset(path=path, name=name, version=version)
    if phase_map:
        for old_phase_name, new_phase_name in phase_map.items():
            replace_key(dataset, old_key=old_phase_name, new_key=new_phase_name)
    if phase_list is None:
        phase_list = dataset.keys()
    examples_dict = {}
    for phase in phase_list:
        phase_examples = []
        for raw_example in dataset[phase]:
            if field_map:
                for old_field_name, new_field_name in field_map.items():
                    replace_key(raw_example, old_key=old_field_name, new_key=new_field_name)
            if label_map and "label" in raw_example:
                # Optionally use an dict or function to map labels
                label = raw_example["label"]
                if isinstance(label_map, dict):
                    if raw_example["label"] in label_map:
                        label = label_map[raw_example["label"]]
                elif callable(label_map):
                    label = label_map(raw_example["label"])
                else:
                    raise TypeError(label_map)
                raw_example["label"] = label
            phase_examples.append(raw_example)
        examples_dict[phase] = phase_examples
    return examples_dict


def write_examples_to_jsonls(examples_dict, task_data_path):
    os.makedirs(task_data_path, exist_ok=True)
    paths_dict = {}
    for phase, example_list in examples_dict.items():
        jsonl_path = os.path.join(task_data_path, f"{phase}.jsonl")
        py_io.write_jsonl(example_list, jsonl_path)
        paths_dict[phase] = jsonl_path
    return paths_dict


def download_and_unzip(url, extract_location):
    """Downloads and unzips a file, and deletes the zip after"""
    _, file_name = os.path.split(url)
    zip_path = os.path.join(extract_location, file_name)
    download_file(url=url, file_path=zip_path)
    unzip_file(zip_path=zip_path, extract_location=extract_location, delete=True)


def unzip_file(zip_path, extract_location, delete=False):
    """Unzip a file, optionally deleting after"""
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(extract_location)
    if delete:
        os.remove(zip_path)


def download_and_untar(url, extract_location):
    """Downloads and untars a file, and deletes the tar after"""
    _, file_name = os.path.split(url)
    tar_path = os.path.join(extract_location, file_name)
    download_file(url, tar_path)
    untar_file(tar_path=tar_path, extract_location=extract_location, delete=True)


def untar_file(tar_path, extract_location, delete=False):
    """Untars a file, optionally deleting after"""
    with tarfile.open(tar_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=extract_location)
    if delete:
        os.remove(tar_path)


def download_file(url, file_path):
    # noinspection PyUnresolvedReferences
    urllib.request.urlretrieve(url, file_path)
