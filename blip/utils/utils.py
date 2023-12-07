"""
Various utility functions
"""
import os
import torch
import copy
import inspect
import shutil
import numpy as np
import pandas as pd
import random
import requests
import tarfile
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from itertools import product
from datetime import datetime


def tar_directory(
    path,
    name
):
    with tarfile.open(name, "w:gz") as tarhandle:
        base_path = os.path.basename(path)
        for root, dirs, files in os.walk(path):
            for f in files:
                file_path = os.path.join(root, f)
                arcname = os.path.relpath(file_path, path)
                tarhandle.add(file_path, arcname=os.path.join(base_path, arcname))


def untar_file(tar_file, output_dir):
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(output_dir)


def index_positions(
    positions,
    indices
):
    return positions[indices]


def get_datetime():
    time = datetime.now()
    now = f"{time.year}.{time.month}.{time.day}.{time.hour}.{time.minute}.{time.second}"
    return now


def get_key(dictionary, val):
    # Given a dictionary and a value, returns list of keys with that value
    res = []
    for key, value in dictionary.items():
        if val == value:
            res.append(key)
    return res


def invert_label_dict(label):
    # Take dict {key:item}, return dict {item:key}
    inverted_label = dict()
    for key in label.keys():
        if isinstance(label[key], list):
            for ll in label[key]:
                inverted_label[ll] = key
        else:
            inverted_label[label[key]] = key
    return inverted_label


def matrix_ell_infinity_distance(M1, M2):
    # Inputs: two numpy arrays of the same size
    # Output: \ell_\infty distance between the matrices
    M = np.abs(M1 - M2)
    dist = np.max(M)
    return dist


def remove_short_bars(barcode, thresh=0.01):
    barcode_thresh = barcode[(barcode[:, 1]-barcode[:, 0] > thresh)]
    return barcode_thresh


def get_array_names(
    input_file: str
):
    """
    Get the names of the arrays in an .npz file
    """
    # check that file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Specified input file: '{input_file}' does not exist!")
    loaded_file = np.load(input_file)
    return list(loaded_file.files)


def flatten_dict(
    dictionary,
    parent_keys=[]
):
    # Flatten the dictionary to get path/value pairs
    items = []
    for key, value in dictionary.items():
        current_keys = parent_keys + [str(key)]
        if isinstance(value, dict):
            items.extend(flatten_dict(value, current_keys))
        else:
            items.append((current_keys, value))
    return items


def generate_combinations_from_arrays(
    dictionary
):
    # Generate combinations from arrays
    arrays = list(dictionary.values())
    all_combinations = list(product(*arrays))
    return all_combinations


def append_npz(
    input_file: str,
    arrays:     dict,
    override:   bool = False,
):
    """
    The following function takes in a .npz file, and a set
    of arrays specified by a dictionary, and appends them
    to the .npz file, provided there are no collisions.
    """
    # check that file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Specified input file: '{input_file}' does not exist!")
    if not isinstance(arrays, dict):
        raise ValueError(f"Specified array must be a dictionary, not '{type(arrays)}'!")
    # otherwise load file and check contents
    loaded_file = np.load(input_file, allow_pickle=True)
    loaded_arrays = {
        key: loaded_file[key] for key in loaded_file.files
    }
    # check that there are no identical array names if override set to false
    if override is False:
        for item in loaded_arrays.keys():
            if item in arrays.keys():
                raise ValueError(f"Array '{item}' already exists in .npz file '{input_file}'!")
    # otherwise add the array and save
    loaded_arrays.update(arrays)
    np.savez(
        input_file,
        **loaded_arrays
    )


def get_method_arguments(method):
    """
    Get a list of arguments and default values for a method.
    """
    # argpase grabs input values for the method
    try:
        argparse = inspect.getfullargspec(method)
        args = argparse.args
        args.remove('self')
        default_params = [None for item in args]
        if argparse.defaults is not None:
            for ii, value in enumerate(argparse.defaults):
                default_params[ii] = value
        argdict = {item: default_params[ii] for ii, item in enumerate(args)}
        return argdict
    except:
        return {}


def get_shape_dictionary(
    dataset=None,
    dataset_loader=None,
    model=None,
):
    """
    Method for getting shapes of data and various other
    useful information.
    """
    data_shapes = {}
    # list of desired dataset values
    dataset_values = [
        'feature_shape',
        'class_shape',
    ]
    for item in dataset_values:
        try:
            data_shapes[item] = getattr(dataset, item)
        except:
            data_shapes[item] = 'missing'
    # list of desired dataloader values
    dataset_loader_values = [
        'num_total_train',
        'num_test',
        'num_train',
        'num_validation',
        'num_train_batches',
        'num_validation_batches',
        'num_test_batches',
    ]
    for item in dataset_loader_values:
        try:
            data_shapes[item] = getattr(dataset_loader, item)
        except:
            data_shapes[item] = 'missing'
    # list of desired model values
    model_values = [
        'input_shape',
        'output_shape',
    ]
    for item in model_values:
        try:
            data_shapes[item] = getattr(model, item)
        except:
            data_shapes[item] = 'missing'
    return data_shapes


def boxcar(
    x,
    mean:   float = 0.11,
    sigma:  float = 0.3,
    mode:   str = 'regular',
):
    """
    Returns a value between -1 and ...
    depending on whether the values (x - (mean +- sigma))
    are < 0, == 0, or > 0.  If
        a) regular :  return 0 if x < or > mean+-sigma
        b) regular :  return 1 if x > low but <= high
    """
    unit = torch.tensor([1.0])
    high = torch.heaviside(x - torch.tensor([mean + sigma]), unit)
    low = torch.heaviside(x - torch.tensor([mean - sigma]), unit)
    if mode == 'regular':
        return low - high
    else:
        return unit + high - low


def get_base_classes(derived):
    """
    Determine the base classes of some potentially inherited object.
    """
    bases = []
    try:
        for base in derived.__class__.__bases__:
            bases.append(base.__name__)
    except:
        pass
    return bases


def generate_plot_grid(
    num_plots,
    **kwargs,
):
    nrows = int(np.floor(np.sqrt(num_plots)))
    ncols = int(np.ceil(num_plots/nrows))
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        **kwargs
    )
    nplots = nrows * ncols
    nextra = nplots - num_plots
    for ii in range(nextra):
        axs.flat[-(ii+1)].set_visible(False)
    return fig, axs


def concatenate_csv(
    files,
    output_file
):
    combined_csv = pd.concat([pd.read_csv(f) for f in files])
    combined_csv.to_csv(output_file, header=None, index=False,)


def get_files(
    directory
):
    return [f for f in listdir(directory) if isfile(join(directory, f))]


def get_files_with_extension(
    root_dir,
    target_extension: str = ''
):
    matching_files = []

    if target_extension != '':
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(target_extension):
                    file_path = os.path.relpath(os.path.join(foldername, filename), root_dir)
                    matching_files.append(file_path)
    else:
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.relpath(os.path.join(foldername, filename), root_dir)
                matching_files.append(file_path)

    return matching_files


def save_model(
    name:   str = '',
    config_file: str = '',
):
    # clean up directories first
    if name == '':
        now = datetime.now()
    else:
        now = name + f"_{datetime.now()}"
    os.makedirs(f"runs/{now}")
    if os.path.isdir("predictions/"):
        shutil.move("predictions/", f"runs/{now}/")
    if os.path.isdir("plots/"):
        shutil.move("plots/", f"runs/{now}/")
    if os.path.isdir("models/"):
        shutil.move("models/", f"runs/{now}/")
    shutil.move(".logs/", f"runs/{now}")
    shutil.move(".checkpoints/", f"runs/{now}")
    shutil.copy(config_file, f"runs/{now}")


def color_list(color):
    '''
    Function which returns the color in ascii.
    '''
    colors = {
        "DEBUG":   '\033[35m',  # PURPLE
        "ERROR":   '\033[91m',  # RED
        "SUCCESS": '\033[92m',  # GREEN
        "WARNING": '\033[93m',  # YELLOW
        "INFO":    '\033[94m',  # BLUE
        "blue":    '\033[94m',  # BLUE
        "magenta": '\033[95m',
        "cyan":    '\033[96m',
        "white":   '\033[97m',
        "black":   '\033[98m',
        "end":     '\033[0m'
    }
    return colors[color]


def download_osf_file(file_url, destination_path):
    response = requests.get(file_url, stream=True)
    with open(destination_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


def print_colored(string, color, bold=False, end="\n"):
    '''
    Print a string in a specific color.
    '''

    color = color_list(color)
    if bold is False:
        output = color + str(string) + color_list("end")
    if bold is True:
        output = '\033[1m' + color + string + color_list("end")

    print(output, end=end)


def update_nested_dictionary_value(
    dictionary,
    keys:   list = [],
    new_value:  any = None
):
    """
    Update the value at the endpoint of a nested dictionary using a list of keys.
    """
    current_dict = dictionary

    # Traverse the dictionary up to the second-to-last level
    for key in keys[:-1]:
        if key in current_dict:
            current_dict = current_dict[key]
        else:
            # Handle the case where the key is not present in the dictionary
            return False

    # Update the value at the endpoint with the new value
    last_key = keys[-1]
    if last_key in current_dict:
        current_dict[last_key] = new_value
        return True
    else:
        # Handle the case where the last key is not present in the dictionary
        return False


def traverse_nested_dictionary(
    dictionary,
    path:   list = None,
    paths:  list = [],
    values: list = []
):
    """
    Traverse every path in a nested dictionary.

    Parameters:
    - d (dict): The nested dictionary to traverse.
    - path (list): The current path being traversed. (Default is None)

    Returns:
    - None
    """
    if path is None:
        path = []

    for key, value in dictionary.items():
        current_path = path + [key]

        if isinstance(value, dict):
            traverse_nested_dictionary(
                value,
                current_path,
                paths,
                values
            )
        else:
            paths.append(current_path)
            values.append(value)

    return paths, values


def generate_random_dictionaries(
    dictionary,
    num_configs:    int = 10,
    sample_dictionary:  dict = None
):
    new_dicts = []

    if sample_dictionary is not None:
        paths, values = traverse_nested_dictionary(sample_dictionary)
    else:
        paths, values = traverse_nested_dictionary(dictionary)

    num_possible_configs = sum([len(x) for x in values])
    num_configs = min(num_possible_configs, num_configs)

    while len(new_dicts) < num_configs:
        sample_dict = copy.deepcopy(dictionary)
        for jj, keys in enumerate(paths):
            update_nested_dictionary_value(
                sample_dict,
                keys,
                random.choice(values[jj])
            )
        if sample_dict not in new_dicts:
            new_dicts.append(sample_dict)

    return new_dicts
