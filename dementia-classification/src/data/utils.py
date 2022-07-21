import os
from hyperpyyaml import load_hyperpyyaml
import subprocess

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
