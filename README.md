dementia-classification
==============================

This project is designed to help diagnose dementia based on spontaneous speech and patients' speech during tasks. The data used in this project can be found in [DementiaBank](https://dementia.talkbank.org/).

Report of this project was rewarded the 3rd place on [HSE conference](https://nnov.hse.ru/studentconf/). 

Attention models and pretrained SASST were used to classify samples of ADReSS 2020 and 2021. 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README about project and achieved results.
    ├── config             <- Certain configs with parameters for data preprocessing and training.
    ├── data
    │   ├── processed      <- Final merged, normalized and denoised audio files for each dataset.
    │   └── raw            <- The original, immutable data dump. It is data from DementiBank and ADReSS challenges
    │   └── speakers_ids.csv   <- ids of all speakers with labels and their corresponding paths.
    │   └── speakers_ids_folded.csv   <- ids of all speakers for speakers in DementiaBank with corresponding three folders train, val, test.
    │
    ├── models             <- Trained and serialized models and their archetectures. 
    │
    ├── notebooks          <- Jupyter notebooks for training, EDA, data visualization and 
    │
    ├── reports            <- Generated analysis as LaTeX
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── dementia_dataset.py - script to get pytorch dataset instance.
    │   │   └── download_save.py - script to download data from Dementia Bank.
    │   │   └── meta.py - script to preprocess cha and csv files in json readable structure.
    │   │   └── process_audio.py - script to denoise, merge and normalize audio files in folders.
    │   │   └── utils.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py

<p><small>Project structure is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.
