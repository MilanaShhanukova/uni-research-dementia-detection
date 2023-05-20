import argparse
import catboost
import pandas as pd
import plotly.express as px
import torch
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.utils import get_task

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report


def save_feature_imp_png(dataset, idx, model, columns_drop, save_dir):
    file_name = dataset.iloc[idx]['file_name']
    dataset = dataset.drop(columns=columns_drop)

    input_data = pd.DataFrame(dataset.iloc[idx].values.reshape(1, -1),
                            columns=dataset.columns)

    pool = catboost.Pool(input_data)  # crate a Pool object for the single sample
    feat_importance = model.get_feature_importance(pool, type=catboost.EFstrType.ShapValues, prettified=True)

    predicted_label = model.predict(pool)

    top_imp_features = torch.topk(torch.tensor(feat_importance.iloc[0].to_list())[:-1], k=5)
    df = pd.DataFrame(dict(
        r=top_imp_features.values.tolist(),
        theta=dataset.columns[top_imp_features.indices.tolist()]))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, )

    fig.update_layout(
    title=f"File {file_name}, predicted label {predicted_label[0]}",
)
    fig.write_image(os.path.join(save_dir, f"{file_name}_{predicted_label[0]}.png"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Validate roberta based model")
    parser.add_argument("--model_file_path")
    parser.add_argument("--reports_folder")
    parser.add_argument("--data_files_dir")
    parser.add_argument("--png_dir")

    parsed_args = parser.parse_args()

    model = CatBoostClassifier()
    model.load_model(parsed_args.model_file_path, format="cbm")
    important_feat = torch.topk(torch.tensor(model.get_feature_importance()), k=5)
    important_feat_names = np.array(model.feature_names_)[important_feat.indices]

    data_golden = pd.read_csv(os.path.join(parsed_args.data_files_dir, "golden_data.csv"))
    data_golden = data_golden.reindex(sorted(data_golden.columns), axis=1)
    data_golden['condition'] = ['Dementia'] * 13
    # golden_drops = ['Unnamed: 0', 'file_name', 'condition', 'text',
    #             'dataset_name', 'fold', 'text.1', ]
    golden_drops = ['target', 'file_name', 'audio_paths', 'dataset_name', 'task', 'condition']
    
    # save png of radar charts
    for idx in range(data_golden.shape[0]):
        save_feature_imp_png(data_golden, idx, model, golden_drops, save_dir=parsed_args.png_dir)
    
    predictions_proba = model.predict_proba(data_golden.drop(columns=golden_drops))
    predictions = model.predict(data_golden.drop(columns=golden_drops))

    data_golden['predicted_label'] = predictions
    data_golden['confidence'] = torch.max(torch.tensor(predictions_proba), dim=1).values

    data_golden['task'] = data_golden['file_name'].apply(get_task)

    with open(os.path.join(parsed_args.reports_folder, 'featured_model_audio_golden.txt'), 'w+') as f:
        print(classification_report(data_golden['condition'], predictions), file=f)

        for task in ["CPD", "Story Recall", "Conversation"]:
            subset = data_golden[(data_golden['task'] == task) & (data_golden['predicted_label'] == 'Dementia')]
            mean_conf = round(subset['confidence'].mean(), 3)
            min_conf = round(subset['confidence'].min(), 3)
            max_conf = round(subset['confidence'].max(), 3)
            f.write(f"{task}: confidence scores stats mean = {mean_conf}, min = {min_conf}, max = {max_conf}")
            f.write('\n')

        f.write(f'Model important features are {important_feat_names}')



    data_golden = data_golden[['file_name', 'task', 'predicted_label', 'confidence']].sort_values(by='file_name')
    data_golden.to_excel(os.path.join(parsed_args.reports_folder, 'data_golden_predictions_featured_audio_model.xlsx'))
    
    
    data_test = pd.read_csv(os.path.join(parsed_args.data_files_dir, "test_data.csv"))
    data_test = data_test.reindex(sorted(data_test.columns), axis=1)
    #test_drops = ['Unnamed: 0', 'file_name', 'condition', 'text', 'text.1', 'task']

    predictions = model.predict(data_test.drop(columns=['target']))


    with open(os.path.join(parsed_args.reports_folder, 'featured_model_audio_test.txt'), 'w+') as f:
        print(classification_report(data_test['target'], predictions), file=f)




