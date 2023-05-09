
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sys
import os
import argparse
import torch

from models.text_roberta_model import CustomSequenceClassification
from sklearn.metrics import classification_report

from src.data.utils import get_task

import pandas as pd
from transformers import AutoTokenizer, Trainer

from src.models.train_roberta_model import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Validate roberta based model")
    parser.add_argument("--model_dir")
    parser.add_argument("--reports_folder")
    parser.add_argument("--text_files_dir")

    parsed_args = parser.parse_args()

    soft = torch.nn.Softmax(dim=1)

    # load the data
    data_test = pd.read_csv(os.path.join(parsed_args.text_files_dir, 'test_texts_emotion_syntax_hesitance.csv'))
    data_golden = pd.read_csv(os.path.join(parsed_args.text_files_dir, 'golden_texts_emotion_syntax_hesitance.csv'))
    data_golden['condition'] = ['Dementia'] * 13

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation=True, max_length=256, padding='max_length')

    dataset_test = prepare_dataset(data_test, tokenizer, mode='test')
    dataset_golden = prepare_dataset(data_golden, tokenizer, mode='test')

    # load the trained model
    trained_model = CustomSequenceClassification.from_pretrained(parsed_args.model_dir,
                                            num_extra_dims=2, local_files_only=True)
    trainer = Trainer(model=trained_model, tokenizer=tokenizer)

    # predict on golden dataset
    predictions = trainer.predict(dataset_golden)

    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    
    probs = soft(torch.tensor(predictions.predictions))
    data_golden['predicted_label'] = predictions.predictions.argmax(1)
    data_golden['confidence'] = torch.max(probs, dim=1).values
    data_golden['task'] = data_golden['file_name'].apply(get_task)
    data_golden = data_golden[['file_name', 'task', 'predicted_label', 'confidence']].sort_values(by='file_name')
    data_golden.to_excel(os.path.join(parsed_args.reports_folder, 'data_golden_predictions_lm_model.xlsx'))

    with open(os.path.join(parsed_args.reports_folder, 'lm_based_model_golden.txt'), 'w+') as f:
        print(classification_report(labels, preds), file=f)
        f.write('\n')

        for task in ["CPD", "Story Recall", "Conversation"]:
            mean_conf = round(data_golden[data_golden['task'] == task]['confidence'].mean(), 3)
            min_conf = round(data_golden[data_golden['task'] == task]['confidence'].min(), 3)
            max_conf = round(data_golden[data_golden['task'] == task]['confidence'].max(), 3)
            f.write(f"{task}: confidence scores stats mean = {mean_conf}, min = {min_conf}, max = {max_conf}")
            f.write('\n')

    # predict on the test dataset
    predictions = trainer.predict(dataset_test)

    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids

    with open(os.path.join(parsed_args.reports_folder, 'lm_based_model_test.txt'), 'w+') as f:
        print(classification_report(labels, preds), file=f)










