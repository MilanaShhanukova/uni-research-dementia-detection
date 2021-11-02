from CHAFile.ChaFile import ChaFile
import json
import os


def parse_header(head_info:list):
    possible_illnesses = ['PPA-NOS']
    clean_info = {'age': 65, 'gender': '', 'condition': 'Dementia'} #need to find the mean age
    if not head_info:
        return None
    for person_info in head_info:
        if 'PAR' not in person_info:
            continue
        person_info = person_info.split('|')
        for info_seg in person_info:
            if info_seg.split(';')[0].isdigit():
                clean_info['age'] = int(info_seg.split(';')[0])
            elif info_seg in ['female', 'male']:
                clean_info['gender'] = info_seg
            elif info_seg in possible_illnesses:
                clean_info['condition'] = info_seg
    return clean_info

def get_meta_info(cha_path: str):
    cha_data = ChaFile(cha_path)
    lines = cha_data.getLinesBySpeakers()['PAR']
    header_info = parse_header(cha_data.speakers_info)
    return [[header_info], [{'text': line['emisión'], 'seconds': line['bullet']} for line in lines if 'bullet' in line]]


def main(dir_path: str, mate_pth: str) -> dict:
    info = {}
    for filename in os.listdir(dir_path):
        info[filename] = get_meta_info(dir_path + "/" + filename)
    with open(mate_pth, "w") as f:
        json.dump(info, f)
