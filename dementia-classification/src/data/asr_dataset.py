from utils import load_config

import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset


class ASRDataset(Dataset):
    """
    Pipeline to load data for asr model.
    """
    def __init__(self, config_path: str, config={}, mode='train'):
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.map_letter2num = config['letter2num']
        self.map_num2letter = {value: key for key, value in self.map_letter2num.items()}
        self.text_transforms = config['text_transform_patterns']

        assert os.path.isfile(config['texts_dataset_path']), 'File with transcriptions was not created'
        self.data = pd.read_csv(config['texts_dataset_path'])

        #transform texts
        self.texts_data = list(map(self.normalize_text, self.data['utterances'].tolist()))

        if mode == 'train':
            self.audio_transforms_operations = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            )
        elif mode == 'val':
            self.audio_transforms_operations = torchaudio.transforms.MelSpectrogram()

    def __len__(self):
        return len(self.texts_data)

    def normalize_text(self, text):
        """
        Normalize text according to the transcription rules.
        :param text:
        :return:
        """
        text = text.lower()
        text = re.sub(r"[^\w\s&+.()']", '', text)

        # graphems
        text = text.replace('&+', ' ')
        text = text.split(' ')

        # double vowels
        for l_idx, l in enumerate(text):
            if l == ':':
                text[l_idx] = text[l_idx-1]
        text = ' '.join(text)

        # other patterns
        for param_name, changes in self.text_transforms.items():
            text = re.sub(changes[0], changes[1], text)
        text = re.sub(r"[^\w\s<>']", '', text)
        return text

    def text_to_num(self, text):
        nums = []
        # text = text.replace(' ', '<SPACE>')
        for i in text:
            try:
                nums.append(self.map_letter2num[i])
            except KeyError:
              pass
        return torch.Tensor(nums)

    def num_to_text(self, nums):
        text = ''.join([self.map_num2letter[num] for num in nums])
        text = text.replace('<SPACE>', ' ')

        return text

    def transform_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        audio = torch.mean(audio, dim=0).unsqueeze(0)

        transform = torchaudio.transforms.Resample(sr, 16000)
        audio = transform(audio)
        spec = self.audio_transforms_operations(audio).squeeze(0).transpose(0, 1)
        return spec

    def collate_func(batch):
        spectrograms = [sample['spectrogram'] for sample in batch]
        texts = [sample['texts'] for sample in batch]

        input_lengths = [spec.shape[0]//2 for spec in spectrograms]
        label_lengths = [len(t) for t in texts]

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(texts, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        texts = self.text_to_num(self.normalize_text(sample['utterances']))
        spec = self.transform_audio(sample['audio_split_path'])

        return {'spectrogram': spec, 'texts': texts}
