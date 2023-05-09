import json
import os
import pandas as pd
import glob
import subprocess
from hyperpyyaml import load_hyperpyyaml


def runcmd(cmd, verbose = False, *args, **kwargs):
    """
    Process wget downloading in the folder.
    """
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)


def get_task(file_name):
    if "cookie" in file_name: return "CPD"
    elif "recall" in file_name: return "Story Recall"
    return "Conversation"


def load_config(config_path: str) -> dict:
    """
    Load configuration file in yaml format.
    :param config_path: full path to configuration file.
    :return: dictionary of all parameters.
    """
    if not os.path.exists(config_path) or not config_path.endswith('.yaml'):
        return {}
    with open(config_path, encoding='utf-8') as f:
        return load_hyperpyyaml(f)

def get_files_names(main_dir, ext):
    """
    Get list of files in one directory and subdirectories.
    :param main_dir: a path to the main directory.
    :param ext: extension of files.
    :return: list of full files paths.
    """
    all_files = []
    for root, _, files in os.walk(main_dir):
        for name in files:
            if name.endswith(ext):
                all_files.append(os.path.join(root, name))
    return all_files


def get_file_name(full_path: str, local_dir: str):
    if '\\' in full_path:
        name = full_path.split('\\')[-1]
    else:
        name = full_path.split('/')[-1]

    try:
        file_name = glob.glob(f'{local_dir}/**/{name}', recursive=True)[0]
        return file_name
    except IndexError:
        return full_path

def filter_data(data_path):
    data = pd.read_csv(data_path)
    data['mode'] = data['audio_paths'].apply(lambda x: x.split('_')[-1][:-4])
    data = data.groupby('mode').agg({'silence_nums': 'mean', 'percent_silence': 'mean'})
    return data

def change_holland_info(json_path, seconds_info):
    with open(json_path) as f:
        data = json.load(f)
    for name, info in data.items():
        for utterance_seconds in info[1]:
            utterance_seconds['seconds'][0] -= seconds_info[name] * 1000
            utterance_seconds['seconds'][1] -= seconds_info[name] * 1000

    with open(json_path[:-5] + '_new.json', 'w') as f:
        json.dump(data, f)

