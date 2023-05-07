import librosa
import torchaudio
import torch
import torchaudio.transforms as T
from tqdm import tqdm
import os


def load_audio(audio_path, target_sr=16000):
    audio, sr = torchaudio.load(audio_path)
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    resampler = T.Resample(sr, target_sr)
    resampled_audio = resampler(audio)
    return resampled_audio


def save_audio_tensor(audio_path: str, audio):
    tensor_path = audio_path.replace('raw', 'audio_tensors')
    if 'mp3' in tensor_path:
        tensor_path = tensor_path.replace('mp3', 'pt')
    elif 'wav' in tensor_path:
        tensor_path = tensor_path.replace('wav', 'pt')
    torch.save(audio, tensor_path)


def save_tensors(audio_raw_folder: str):
    unsaved_paths = []
    audio_tensors_path = audio_raw_folder.replace('raw', 'audio_tensors')
    if not os.path.exists(audio_tensors_path):
        os.makedirs(audio_tensors_path)

    audio_paths = [f for f in os.listdir(audio_raw_folder) if f.endswith('mp3') or f.endswith('wav')]
    for audio_path in tqdm(audio_paths):
        full_audio_path = os.path.join(audio_raw_folder, audio_path)
        try:
            resampled_audio = load_audio(full_audio_path)
            save_audio_tensor(full_audio_path, resampled_audio)
        except RuntimeError:
            unsaved_paths.append(audio_path)
    return unsaved_paths
