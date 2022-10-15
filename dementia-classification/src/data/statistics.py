import json
import glob
import re
import os
from typing import Dict, List
from collections import Counter
import pandas as pd
import tqdm

from utils import load_config

# Holland - split by yourself when certain tasks begins
# Kempler - convs, one d6 cookie
# Hopkins - problems with downloading
# Lanzi - convs
# Pitt - cookie, fluency, recall
# WLS - cookie theft description
# DePaul - long interview


info_tasks = {
    'Kempler': ['conversation', 'cookie'],
    'Lanzi': ['conversation'],
    'Pitt': ['cookie', 'fluency', 'recall'],
    'WLS': ['cookie']
}

patterns = {
    'num_short_pauses': r'\(\.\)', 'num_long_pauses': r'\(\.\.\)',
    'num_very_long_pauses': r'\(\.\.\.\)', 'num_fillers': r'&[-\w]\w+', 'num_unintelligent_word': r'xxx'
}



def parse_dataset(dataset_name: str, folder_path: str, json_path: str, unique_fillers: Dict,
                  patterns={}, info_tasks={}) ->  (Dict, List[set]):

    data = {
            'name': [], 'text': [], 'condition': [], 'task': [],
            'num_short_pauses': [], 'num_long_pauses': [], 'num_very_long_pauses': [],
            'num_fillers': [], 'num_unintelligent_word': []
        }

    with open(json_path) as f:
        data_raw = json.load(f)

        for name, info in data_raw.items():
            data['name'].append(name)
            try:
                data['condition'].append(info[0][0]['condition'])
            except IndexError:
                data['condition'].append(' ')
            # get task from path
            data['task'].append(get_task(dataset_name, name, folder_path, info_tasks))
            whole_utterance = ' '.join([_['text'] for _ in info[1]])
            data['text'].append(whole_utterance)

            # count all patterns
            for pattern_name, pattern in patterns.items():
                pattern_frequencies = re.findall(pattern, whole_utterance)
                data[pattern_name].append(len(pattern_frequencies))

                if pattern_name == 'num_fillers':
                    unique_fillers.update(pattern_frequencies)

    return data, unique_fillers


def get_task(dataset_name: str, name, folder_path: str, info_tasks: Dict) -> str:
    task = info_tasks[dataset_name]
    if len(task) > 1:
        try:
            audio_path = glob.glob(f'{folder_path}/**/{name}.mp3', recursive=True)[0]
            for t in task:
                if t in audio_path:
                    task = t
                    return task
        except IndexError:
            assert FileExistsError
    return task[0]


def common_save_stats(config_path: str, datasets_names=['Pitt', 'Kempler', 'Lanzi', 'WLS']) -> Dict:
    config = load_config(config_path)
    raw_path = config['raw_path']
    info_tasks = config['info_tasks']
    patterns = config['patterns']
    unique_fillers = Counter()

    for d_name in tqdm.tqdm(datasets_names):
        data, unique_fillers = parse_dataset(d_name, os.path.join(raw_path, d_name),
                                             os.path.join(raw_path, d_name + '.json'), unique_fillers,
                                             patterns, info_tasks)
        pd.DataFrame.from_dict(data).to_csv(os.path.join(raw_path, f'statistics_{d_name}.csv'))

    return unique_fillers
