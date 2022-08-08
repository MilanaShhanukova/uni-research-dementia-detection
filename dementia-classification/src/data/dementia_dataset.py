from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import torch
import torchaudio
from utils import load_config, get_files_names


class AudioDatasetExternal(Dataset):
    """
    Pipeline to load and featurize audio data.
    """
    def __init__(self, config_path: str, config={}):
        """
        Initialize main parameters.
        :param config_path: full path to yaml configuration.
        :param config: dictionary of parameters if they are changed in training pipeline manualy.
        """
        # if configuration manualy changed as a dictionary
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.label2num = {'Dementia': 1, 'Control': 0}

        # spectrograms normalization parameters
        self.normalize_mode = self.config['normalize']
        self.mean, self.std = self.config['mean'], self.config['std']


        self.p2db = self.config['p2db']
        self.p2b_transformer = torchaudio.transforms.AmplitudeToDB() if self.config['p2db'] else None # turn power to db

        # modes of datasets
        self.dataset_type = self.config['dataset_type'] # dementia bank, adress-2020, adress-2021, adress-2020-2021
        self.mode = self.config['mode'] # train - val - test - all
        self.fold = self.config['fold'] # for dementia bank if train mode use different preprocessed folds
        self.dataset_type = self.config['dataset_type']

        # certain paths and their labels
        self.speakers_ids = pd.read_csv(self.config['speakers_ids_path'])

        # preprocess data
        self.features, self.labels = self.get_data()

    def get_data(self):
        all_features, all_labels = [], []

        if 'adress' in self.dataset_type:
            idx = f"adress_path_{self.mode}"
            audio_paths = pd.read_csv(self.config[idx])['merged_audio_path'].to_list()
        else:
            audio_paths = self.speakers_ids[self.speakers_ids[self.fold] == self.mode]['merged_audio_path'].to_list()

        feature_extractor = self.config[f'{self.config["data_type"]}_extractor']
        for audio_p in audio_paths:
            label = self.speakers_ids[self.speakers_ids['merged_audio_path'] == audio_p]['label'].item()
            label = self.label2num[label]
            frames = self.split_audio_by_frames(audio_p, self.config)

            features = [feature_extractor(frame) for frame in frames if frame.shape[1] > 1]

            # normalization
            if self.normalize_mode:
                features = [self.normalize(f, self.mean, self.std) for f in features]
            # power to bd
            if self.p2db:
                features = [self.p2b_transformer(f) for f in features]

            all_features.extend(features)
            all_labels.extend([label] * len(features))
        return all_features, all_labels

    @staticmethod
    def split_audio_by_frames(audio_path, config):
        audio, sr = torchaudio.load(audio_path, config['sr'])
        audio = torch.mean(audio, 0).unsqueeze(0)

        split_size_samples = int(config['sr'] * config['per_second'])
        chunks = list(torch.split(audio, split_size_samples, dim=-1))

        # do padding
        for idx, chunk in enumerate(chunks):
            if chunk.shape[1] != chunks[0].shape[1]:
                shortage = config['sr'] * config['per_second'] - chunk.shape[1]
                chunks[idx] = torch.nn.functional.pad(chunk, (0, shortage))
        return chunks

    def normalize(self, feature, mean, std):
        return feature * std + mean

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'label': self.labels[idx]}
