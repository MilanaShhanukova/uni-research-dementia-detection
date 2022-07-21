from utils import runcmd, get_files_names, load_config

def normalize_audio_pipeline(config_path):
    config = load_config(config_path)

    raw_files = get_files_names(config['raw_data_main_dir'], ext='mp3')

    for raw_f in raw_files:
        # normalize
        normalize_path = raw_f.replace('audio', 'norm_audio')
        runcmd(f'ffmpeg-normalize {raw_f} -ar 16000 -nt rms -c:a libmp3lame -e ="-ac 1" -f -q -o {normalize_path}')
    for dataset_name config['dataset_names']:

    # denoize



preprocess_audio_pipeline(r'C:\Users\Милана\PycharmProjects\course_work\dementia-classification\configs\constant_preprocess.yaml')


