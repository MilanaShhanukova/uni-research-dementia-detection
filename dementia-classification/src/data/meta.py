import json
import os
import sys
sys.path.insert(0, r'C:\Users\Милана\PycharmProjects\course_work\CHAFile')
from CHAFile.ChaFile import ChaFile
from utils import load_config, get_files_names


class ChaPreprocesser:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def parse_header(self, head_info: list, base_cond: str) -> dict:
        """
        Get basic one sample information of age, gender, condition.
        :param head_info: information from cha preprocessing.
        :param base_cond: condition of the folder.
        :return: basic parameters of one sample.
        """
        clean_info = {'age': 65, 'gender': '', 'condition': base_cond}
        if not head_info:
            return {'condition': base_cond}
        for person_info in head_info:
            person_info = person_info.split('|')
            for idx, info_seg in enumerate(person_info):
                if info_seg.split(';')[0].isdigit() and idx != len(person_info) - 1:
                    clean_info['age'] = int(info_seg.split(';')[0])
                elif info_seg in ['female', 'male']:
                    clean_info['gender'] = info_seg
                elif info_seg in self.config['illnesses']:
                    clean_info['condition'] = info_seg
                elif info_seg.isdigit():
                    clean_info['participant'] = int(info_seg)
        return clean_info

    def get_meta_info(self, cha_path: str, base_cond:'Dementia') -> list:
        """
        Get basic info of one sample and its lines with certain seconds.
        :param cha_path: full path to cha info.
        :param base_cond: condition of one file if not mentioned in file.
        :return: basic parameters and lines of one speaker.
        """
        cha_data = ChaFile(cha_path)
        lines = cha_data.getLinesBySpeakers()
        if not lines:
            return [[], []]
        lines = lines['PAR']
        header_info = self.parse_header(cha_data.speakers_info, base_cond)
        return [[header_info], [dict(text=line['emisión'], seconds=line['bullet'])
                                for line in lines if 'bullet' in line]]

    def main(self) -> dict:
        """
        Preprocess all cha files in all directory to one json file.
        Get rid of other unuseful parameters and filter out lines of participant.
        :return:
        """
        for dataset_name, base_cond in self.config['datasets_info'].items():
            info = {}
            cha_folder_path = self.make_path(self.config, dataset_name, 'cha_folder')
            cha_paths = get_files_names(cha_folder_path, '.cha')
            for cha_p in cha_paths:
                short_name = cha_p.split("\\")[-1][:-4]
                info[short_name] = self.get_meta_info(cha_p, base_cond)

            save_path = self.make_path(self.config, dataset_name, 'json')
            print(save_path)
            with open(save_path, "w") as f:
                json.dump(info, f)

    @staticmethod
    def make_path(config: dict, dataset_name: str, mode='json') -> str:
        """
        Make paths for json files and meta folders.
        :param config: configuration as a dict.
        :param dataset_name: name of main dataset, e.g. Lanzi.
        :param mode: type of path to create, e.g. json, meta folder.
        :return:
        """
        dir_path = config['raw_data_main_dir']
        if mode == 'json':
            path_ = os.path.join(dir_path, dataset_name, dataset_name + '.json')
            return path_
        elif mode == 'cha_folder':
            path_ = os.path.join(dir_path, dataset_name, 'meta_cha')
            return path_
