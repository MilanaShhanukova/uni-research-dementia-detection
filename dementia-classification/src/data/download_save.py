from utils import load_config, runcmd
import os


def download_pipeline(config_path: str, user: str, password: str, main_dir=r'C:\Users\Милана\PycharmProjects\course_work\dementia-classification\data\raw'):
    """
    Download and save meta data and audio for datasets names in folders.
    :param config_path: full path to configuration file with raw data path and names of datasets
    :param user: name of the user to get data
    :param password: password to get data
    """
    # for all names in dataset names do downloading and saving
    config = load_config(config_path)

    for d_name in config['datasets_names']:
        print(f'Start downloading and saving dataset {d_name}')
        save_path_folder = generate_save_path(main_dir, d_name)

        #download and save audio
        download_path_audio = generate_download_path(d_name, '')
        save_path_audio = generate_save_path(main_dir, d_name, 'audio')
        runcmd(f'wget --http-user={user} --http-password={password} -q -r -np -nH --cut-dirs=2 -A *.mp3 {download_path_audio} -P {save_path_audio}')

        # generate and save meta files
        download_path_meta_zip = generate_download_path(d_name, 'zip')
        save_path_meta = generate_save_path(main_dir, d_name, 'meta_cha')
        runcmd(f'wget --http-user={user} --http-password={password} {download_path_meta_zip} -P {save_path_folder}')
        runcmd(f'unzip {save_path_folder + d_name + ".zip"} -d {save_path_meta}')


def generate_save_path(main_dir: str, dataset_name:str, data_type=''):
    """
    Makes a directory for certain dataset and its type. e.g. 'dementia/Lanzi/audio'.
    :param main_dir: full path to all folders.
    :param dataset_name: name of the main folder, e.g. Lanzi.
    :param data_type: name of subfolder e.g. audio, text, json.
    :return: create a new directory.
    """
    folder_path = os.path.join(main_dir, dataset_name, data_type)

    if os.path.isdir(folder_path):
        print(f'Folder {folder_path} already exists')
    else:
        os.mkdir(folder_path)
        print(f'Folder {folder_path} was created')
    return folder_path


def generate_download_path(dataset_name: str, data_type='') -> str:
    """
    Creates a path to download dementia bank dataset.
    :rtype: object
    :param dataset_name: name of folder, e.g. Lanzi
    :param data_type: zip type or not, e.g. zip
    :return: full path to download files
    """
    main_dir = 'https://media.talkbank.org/dementia/English/'
    if data_type == 'zip':
        main_dir = 'https://dementia.talkbank.org/data/English/'
        dataset_name += '.zip'
        download_path = main_dir + dataset_name
        return download_path

    download_path = main_dir + dataset_name + '/'
    return download_path

if __name__ == '__main__':
    # download
    print('Start downloading files')
    download_pipeline(r'C:\Users\Милана\PycharmProjects\course_work\dementia-classification\configs\constant_preprocess.yaml', 'broca', 'wernicke')
    # get audio from video

