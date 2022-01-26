import json
import os

import librosa
import torch
import torch.nn.functional as F
from pydub import AudioSegment
import noisereduce as nr


def load_meta_info(meta_path):
    with open(meta_path) as meta_file:
        meta_data = json.load(meta_file)
    return meta_data


# first case - split by duration in cha files
def split_meta_info(meta_info_path, audio_raw_path, audio_split_path):
    """
    Split data according to json meta data
    :param meta_info_path: path to a json file
    :param audio_raw_path: path to a dir with raw not splitted audio
    :param audio_split_path: path to a dir for new splitted audio
    """
    meta_info = load_meta_info(meta_info_path)
    clean_info = []

    for audio_name in meta_info:
        audio_path = os.path.join(audio_raw_path, audio_name[:-4] + '.mp3')
        wave = AudioSegment.from_mp3(audio_path)
        condition = meta_info[audio_name][0][0]['condition']

        for record_info in meta_info[audio_name][1]:
            start, end = record_info['seconds'][0], record_info['seconds'][1]
            fragment_path = os.path.join(audio_split_path, f'{audio_name[:-4]}_{condition}_{start}.wav')
            fragment_audio = wave[start:end]

            os.chdir(audio_split_path)
            fragment_audio.export(f'{audio_name[:-4]}_{condition}_{start}.wav', format='wav')

            record_info['fragment_name'] = fragment_path
            record_info['condition'] = condition

            clean_info.append(record_info)

    return clean_info


def merge_audio_files(meta_info_path, audio_raw_path, audio_merged_path):
    """
    Merge data according to meta info.
    :param meta_info_path: path to a json file
    :param audio_raw_path: path to a dir with raw audio
    :param audio_split_path: path to a dir for new merged data
    """
    meta_info = load_meta_info(meta_info_path)
    clean_info = []

    for audio_name in meta_info:
        audio_path = os.path.join(audio_raw_path, audio_name[:-4] + '.mp3')
        condition = meta_info[audio_name][0][0]['condition']

        wave = AudioSegment.from_mp3(audio_path)
        clean_audio = []

        for record_info in meta_info[audio_name][1]:
            start, end = record_info['seconds'][0], record_info['seconds'][1]

            clean_audio.append(wave[start:end])

        clean_audio = sum(clean_audio)
        clean_audio_path = os.path.join(audio_merged_path, f'{audio_name[:-4]}_.wav')
        clean_audio.export(clean_audio_path, format='wav')

        record_info['condition'] = condition
        clean_info.append(record_info)
    return clean_info


# second case - split by frames
def split_audios_by_frames(audio_path, sr=44100, split_size_seconds=10):
    """
    Split audio file according by seconds - 5 seconds
    :param audio_path - the path of one audio fragments, supposed to be long one.
    """
    wav, sr = librosa.load(audio_path)

    # reduce noise
    wav = nr.reduce_noise(wav, 44100)
    wav = torch.Tensor(wav).unsqueeze(0)

    wav = torch.mean(wav, 0)
    split_size_seconds = 10
    split_size_samples = int(sr * split_size_seconds)
    chunks = list(torch.split(wav, split_size_samples, dim=-1))

    # do padding
    for idx, chunk in enumerate(chunks):
        if chunk.shape[0] != chunks[0].shape[0]:
            shortage = split_size_seconds * sr - chunk.shape[0]
            chunks[idx] = torch.nn.functional.pad(chunk, (0, shortage))
    return chunks


def get_spec(wav, sr=44100):
    melspec = librosa.feature.melspectrogram(wav.numpy(), sr)
    return torch.Tensor(melspec).unsqueeze(0)


def get_all_specs(folders):
    specs = []

    for folder in folders:
        paths = [folder + '/' + path for path in os.listdir(folder)]
        for path in paths:
            if not path.endswith('.wav'):
                continue
            chunks = split_audios_by_frames(path)
            local_specs = [get_spec(chunk).unsqueeze(0) for chunk in chunks]
            specs += local_specs
    return specs
