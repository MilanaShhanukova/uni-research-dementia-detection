from utils import runcmd, get_files_names, load_config

import json
import glob
import os
import pandas as pd
import moviepy.editor as mp
from typing import Dict
from pydub import AudioSegment
from tqdm import tqdm


class AudioPreprocesser:
    def __init__(self, config_path: str, modes=['normalize', 'denoise', 'merge', 'split']):
        self.config = load_config(config_path)
        self.raw_path = self.config['raw_data_main_dir'] # if used too often make a variable
        self.modes = modes

        self.speakers_ids = pd.read_csv(self.config['speakers_database'])

    def split_audio_files(self, json_meta_path: str) -> Dict:
        """
        Split audio files by seconds and save in a corresponding directory. For all saved audio fragments save text
        utterances in one csv file.
        :param json_meta_path: whole path to the json file with seconds and transcriptions.
        :return:
        """
        splitted_info = {'audio_split_path': [], 'utterances': []}
        meta_info = self.load_meta_info(json_meta_path)
        for audio_name in meta_info:
            try:
                audio_path = glob.glob(self.raw_path + "\\**\\" + audio_name + '.wav', recursive=True)[0]
            except IndexError:
                continue

            save_audio_path = audio_path.replace('raw', 'asr_processed')
            save_dir = os.path.dirname(save_audio_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # split audio by provided seconds
            wave = AudioSegment.from_wav(audio_path)
            audio_info = meta_info[audio_name][1]
            if len(audio_info) == 0:
                continue

            for record_info in audio_info:
                start, end = record_info['seconds'][0], record_info['seconds'][1]

                segment_path = f'{save_audio_path[:-4]}_{start}_{end}_.wav'

                wave[start:end].export(segment_path, format='wav')
                splitted_info['audio_split_path'].append(segment_path)
                splitted_info['utterances'].append(record_info['text'])
        pd.DataFrame().from_dict(splitted_info).to_csv(json_meta_path.replace('json', 'csv'))
        return splitted_info

    def merge_audio_files(self, json_meta_path: str):
        """
        Merged data according to json file and saves it in wav format
        :param json_meta_path: full math to one dataset meta info path
        """
        meta_info = self.load_meta_info(json_meta_path)
        for audio_name in meta_info:
            try:
                audio_path = glob.glob(self.raw_path + "\\**\\" + audio_name + '.wav', recursive=True)[0]
            except IndexError:
                continue

            save_audio_path = audio_path.replace('raw', 'processed')
            save_dir = os.path.dirname(save_audio_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # join audio by provided seconds
            wave = AudioSegment.from_wav(audio_path)
            audio = []
            audio_info = meta_info[audio_name][1]
            if len(audio_info) == 0:
                continue
            for record_info in audio_info:
                start, end = record_info['seconds'][0], record_info['seconds'][1]
                audio.append(wave[start:end])
            audio = sum(audio)
            audio.export(save_audio_path, format='wav')
            dataset_name = [n for n in self.config['datasets_names'] if n in save_audio_path][0]
            diagnose = self.get_diagnose(self.config, dataset_name, save_audio_path)
            audio_params = {'speaker_id': self.speakers_ids.shape[0], 'label': diagnose,
                            'audio_path': audio_path, 'merged_audio_path': save_audio_path}
            self.speakers_ids = self.speakers_ids.append(audio_params, ignore_index=True)
        self.speakers_ids.to_csv(self.config['speakers_database'], index=False)

    @staticmethod
    def load_meta_info(meta_path) -> dict:
        """
        Load json file
        :param meta_path: path to json file
        :return: json dict
        """
        with open(meta_path) as meta_file:
            meta_data = json.load(meta_file)
        return meta_data


    @staticmethod
    def get_diagnose(config, dataset_name, audio_path) -> str:
        """
        Get diagnose from either dataset name or audio path.
        :param config: configuration for dataset maintaining.
        :param dataset_name: name of processed dataset.
        :param audio_path: full audio path of processed file.
        :return:
        """
        diagnose = config['datasets_info'][dataset_name]
        if diagnose == 'both':
            diagnose = 'Control' if 'Control' in audio_path or 'cn' in audio_path else 'Dementia'
        return diagnose

    def normalize(self):
        for dataset_name in tqdm(self.config['datasets_names']):
            if self.config['dataset_type'] != 'adress':
                raw_files_dir = os.path.join(self.config['raw_data_main_dir'], dataset_name, 'audio')
            else:
                raw_files_dir = self.config['raw_data_main_dir']
            raw_files = get_files_names(raw_files_dir, ext='wav')
            for raw_f in tqdm(raw_files):
                # normalize
                normalize_path = raw_f.replace('audio', 'norm_audio')
                if 'mp3' in normalize_path:
                    normalize_path = normalize_path.replace('mp3', 'wav')
                if not os.path.exists(os.path.dirname(normalize_path)):
                    os.mkdir(os.path.dirname(normalize_path))
                runcmd(
                    f'ffmpeg-normalize {raw_f} -ar 16000 -nt rms -c:a libmp3lame -e="-ac 1" -f -q -o {normalize_path}')
                sound = AudioSegment.from_mp3(normalize_path)
                sound.export(normalize_path, format="wav")

    def denoise(self):
        for dataset_name in tqdm(self.config['datasets_names']):
            # denoise
            noisy_dir = os.path.join(self.config['raw_data_main_dir'], dataset_name, 'norm_audio', dataset_name)
            out_dir = os.path.join(self.config['raw_data_main_dir'], dataset_name, 'clean_audio')
            runcmd(
                f'python3 -m denoiser.enhance --master64 --dry 0.2 --noisy_dir={noisy_dir} --out_dir={out_dir} --streaming')

    @staticmethod
    def get_audio_from_video(video_folder_path):
        """
        Get audio in mp3 format from video.
        :param video_folder_path: full path to the folder with mp4 video files.
        """
        video_paths = get_files_names(video_folder_path, 'mp4')
        for video_p in tqdm(video_paths):
            video = mp.VideoFileClip(video_p)
            video.audio.write_audiofile(video_p[:-1] + '3')

    def main(self,):
        """
        Main function to save files.
        :return: loaded spectrograms
        """
        if 'normalize' in self.modes:
            self.normalize()
        if 'denoise' in self.modes:
            self.denoise()

        json_paths = glob.glob(self.raw_path + "/**/*.json")
        if not json_paths:
            json_paths = glob.glob(self.raw_path + '/*.json')

        for json_p in tqdm(json_paths):
            if 'merge' in self.modes:
                self.merge_audio_files(json_p)
            if 'split' in self.modes:
                self.split_audio_files(json_p)
