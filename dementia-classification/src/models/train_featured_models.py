import pandas as pd

from catboost import CatBoostClassifier, metrics
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Training featured based model")
    parser.add_argument("--data_files_dir")
    parser.add_argument("--file_save_name")

    parsed_args = parser.parse_args()

    data_train = pd.read_csv(os.path.join(parsed_args.data_files_dir, "train_data.csv"))
    data_test = pd.read_csv(os.path.join(parsed_args.data_files_dir, "test_data.csv"))
    data_golden = pd.read_csv(
        os.path.join(parsed_args.data_files_dir, "golden_data.csv")
    )
    data_train = data_train.reindex(sorted(data_train.columns), axis=1)
    data_test = data_test.reindex(sorted(data_test.columns), axis=1)    

    model = CatBoostClassifier(
        custom_loss=[metrics.Accuracy()], random_seed=24, logging_level="Silent"
    )

    model.fit(
        data_train.drop(columns=["Unnamed: 0", "condition", "file_name", "task", "text", "text.1"]),
        data_train["condition"],
        eval_set=(
            data_test.drop(columns=["Unnamed: 0", "condition", "file_name", "task", "text", "text.1"]),
            data_test["condition"],
        ),
        plot=True,
    )

    model.save_model(parsed_args.file_save_name, format="cbm", export_parameters=None, pool=None)
