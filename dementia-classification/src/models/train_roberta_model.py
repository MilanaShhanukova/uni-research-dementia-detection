
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from models.text_roberta_model import CustomSequenceClassification
from transformers import TrainingArguments, Trainer, AutoTokenizer
import argparse
import datasets
import os
import pandas as pd


def create_example(row):
    text = row['text']
    extra_data = row[[ 'polarity', 'subjectivity']].tolist()
    label = row['condition']
    return {
        'text': text,
        'extra_data': extra_data,
        'label': label
    }


def prepare_dataset(data, tokenizer, mode='train'):
    data['condition'] = data['condition'].apply(lambda x: 1 if x == 'Dementia' else 0)

    samples = data.apply(create_example, axis=1).tolist()
    dataset = datasets.Dataset.from_dict({'text': [i['text'] for i in samples],
                                            'extra_data': [i['extra_data'] for i in samples],
                                            'label': [i['label'] for i in samples],})
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding='max_length'))
    dataset = dataset.remove_columns(["text"])

    if mode == 'train':
        dataset = dataset.shuffle(seed=42)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Training roberta based model')
    parser.add_argument('--text_files_dir')
    parser.add_argument('--save_dir')

    parsed_args = parser.parse_args()

    data_train = pd.read_csv(os.path.join(parsed_args.text_files_dir, 'train_texts_emotion_syntax_hesitance.csv'))
    data_test = pd.read_csv(os.path.join(parsed_args.text_files_dir, 'test_texts_emotion_syntax_hesitance.csv'))
    data_golden = pd.read_csv(os.path.join(parsed_args.text_files_dir, 'golden_texts_emotion_syntax_hesitance.csv'))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation=True, max_length=256, padding='max_length')

    # TRAIN
    dataset_train = prepare_dataset(data_train, tokenizer)
    dataset_test = prepare_dataset(data_test, tokenizer, mode='test')
    dataset_golden = prepare_dataset(data_golden, tokenizer, mode='test')

    model_classifier = CustomSequenceClassification.from_pretrained("roberta-base", num_labels=2, num_extra_dims=2)

    args = TrainingArguments(output_dir="./", weight_decay=0.001, num_train_epochs=5,
                            use_mps_device=True)

    trainer = Trainer(model=model_classifier, train_dataset=dataset_train, tokenizer=tokenizer, args=args)

    trainer.train()
    trainer.save_model(parsed_args.save_dir)
