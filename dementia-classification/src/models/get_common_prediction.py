import pandas as pd
import plotly.express as px
import os
import numpy as np


def save_common_res_png(file_name, scores, models_names, save_dir):
    df = pd.DataFrame(dict(
        r=scores,
        theta=models_names))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, )

    fig.update_layout(
    title=f"File {file_name}",
)
    fig.write_image(os.path.join(save_dir, f"{file_name}.png"))


if __name__ == '__main__':
    main_dir = ''
    save_dir = ''
    audio_feat = pd.read_csv(os.path.join(main_dir, 'golden_audio_featured.csv'))
    audio_att = pd.read_csv(os.path.join(main_dir, 'golden_att_audio.csv'))
    lm_feat = pd.read_csv(os.path.join(main_dir, 'golden_lm_roberta.csv'))
    text_feat = pd.read_csv(os.path.join(main_dir, 'golden_text_featured.csv'))


    for idx in range(13):
        feats = [audio_att.loc[idx, 'confidence'], audio_feat.loc[idx, 'confidence'],
                 text_feat.loc[idx, 'confidence'], lm_feat.loc[idx, 'confidence']]
        names = ['Audio attention', 'Audio featured', 'Text featured', 'Text LM']

        file_name = lm_feat.loc[idx, 'file_name']

        predicted_label = 'Contol'
        if np.mean(feats) > 0.5:
            predicted_label = 'Dementia'
        
        save_common_res_png(file_name, feats, names, predicted_label, save_dir)
        