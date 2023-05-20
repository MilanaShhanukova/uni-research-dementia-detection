import pandas as pd
import plotly.express as px
import os
import numpy as np
import plotly.graph_objects as go


def save_common_res_png(file_name, scores, models_names, save_dir):
    df = pd.DataFrame(dict(
        r=scores,
        theta=models_names))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, )

    fig.update_layout(
    title=f"File {file_name}",
)
    fig.write_image(os.path.join(save_dir, f"{file_name}.png"))


def get_task_depended_visualisation(save_dir, speakers, text_feat, audio_att, audio_feat, lm_feat):

    names = ['Audio attention', 'Audio featured', 'Text featured', 'Text LM', 'Audio attention']

    for speaker in set(speakers):
        fig = go.Figure()

        tasks = text_feat[text_feat['speaker'] == speaker]['task'].to_list()

        for task in tasks:
            audio_att_subset = audio_att[(audio_att['speaker'] == speaker) & (audio_att['task'] == task)]

            audio_feat_subset = audio_feat[(audio_feat['speaker'] == speaker) & (audio_feat['task'] == task)]

            text_feat_subset = text_feat[(text_feat['speaker'] == speaker) & (text_feat['task'] == task)]

            lm_feat_subset = lm_feat[(lm_feat['speaker'] == speaker) & (lm_feat['task'] == task)]


            feats = [audio_att_subset.iloc[0]['confidence'],
                    audio_feat_subset.iloc[0]['confidence'],
                    text_feat_subset.iloc[0]['confidence'],
                    lm_feat_subset.iloc[0]['confidence'],
                    audio_att_subset.iloc[0]['confidence']]


            file_name = lm_feat.loc[idx, 'file_name']
            
            fig.add_trace(go.Scatterpolar(
            r=feats,
            theta=names,
            name=task,
            ))

        fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=True,
        title=speaker
        )
        
        fig.write_image(os.path.join(save_dir, f"{speaker}.png"))
    


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
