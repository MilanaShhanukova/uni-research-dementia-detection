from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import torch
import torchaudio
from utils import load_config, get_files_names



class AudioDatasetExternal(Dataset):
    def __init__(self, config_path, dataset_type, fold='fold_1', mode='train'):
        self.config = load_config(config_path)

        self.mode = mode
        self.fold = fold
        self.dataset_type = dataset_type

        self.speakers_ids = pd.read_csv(self.config['speakers_ids_path'])
        self.features, self.labels = self.get_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'label': self.labels[idx]}

    def get_data(self):
        all_features, all_labels = [], []
        if self.dataset_type == 'adress':
            audio_paths = get_files_names(self.config['adress_path'], 'wav')
        else:
            audio_paths = self.speakers_ids[self.speakers_ids[self.fold] == self.mode]['merged_audio_path'].to_list()
        feature_extractor = self.config[f'{self.config["data_type"]}_extractor']

        for audio_p in audio_paths:
            if self.dataset_type == 'adress':
                label = 'Dementia' if 'cd' in audio_p else 'Control'
            else:
                label = self.speakers_ids[self.speakers_ids['merged_audio_path'] == audio_p]['label'].item()

            frames = self.split_audio_by_frames(audio_p, self.config)

            features = [feature_extractor(frame) for frame in frames if frame.shape[1] > 1]
            if not features:
                continue
            all_features.extend(features)
            all_labels.extend([label] * len(features))
        return all_features, all_labels

    @staticmethod
    def split_audio_by_frames(audio_path, config):
        audio, sr = torchaudio.load(audio_path, config['sr'])

        split_size_samples = int(config['sr'] * config['per_second'])
        chunks = list(torch.split(audio, split_size_samples, dim=-1))

        # do padding
        for idx, chunk in enumerate(chunks):
            if chunk.shape[1] != chunks[0].shape[1]:
                shortage = config['sr'] * config['per_second'] - chunk.shape[1]
                chunks[idx] = torch.nn.functional.pad(chunk, (0, shortage))
        return chunks



train_dataset = AudioDatasetExternal(r'C:\Users\Милана\PycharmProjects\course_work\dementia-classification\configs\train_spectrogram_example.yaml',
                                     'adress')

dataloader = DataLoader(train_dataset, batch_size=4)
print(next(iter(dataloader)))