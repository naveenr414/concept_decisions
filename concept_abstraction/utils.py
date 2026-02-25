import numpy as np
import glob
import ujson as json
import os


def get_save_path(folder_name, result_name):
    """Return the path to save a result JSON file.

    Args:
        folder_name: Subfolder within results/, e.g. 'basic'
        result_name: Filename prefix

    Returns:
        String path ending in .json
    """
    return "{}/{}.json".format(folder_name, result_name)


def delete_duplicate_results(folder_name, result_name, data):
    """Delete any existing result files whose parameters match data['parameters'].

    Args:
        folder_name: Subfolder within results/
        result_name: Filename prefix to glob
        data: Dict containing a 'parameters' key
    """
    all_results = glob.glob("../results/{}/{}*.json".format(folder_name, result_name))

    for file_name in all_results:
        try:
            with open(file_name) as f:
                load_file = json.load(f)
            if load_file.get("parameters") == data["parameters"]:
                os.remove(file_name)
        except Exception as e:
            print(f"Warning: could not process {file_name}: {e}")


def get_results_matching_parameters(folder_name, result_name, parameters):
    """Return all result dicts whose parameters are a superset of `parameters`.

    Args:
        folder_name: Subfolder within results/
        result_name: Filename prefix to glob
        parameters: Dict of key/value pairs that must all match

    Returns:
        List of result dicts
    """
    all_results = glob.glob("results/{}/{}*.json".format(folder_name, result_name))
    ret_results = []

    for file_name in all_results:
        try:
            with open(file_name) as f:
                load_file = json.load(f)
        except Exception:
            continue

        file_params = load_file.get("parameters", {})
        if all(file_params.get(k) == v for k, v in parameters.items()):
            ret_results.append(load_file)

    return ret_results