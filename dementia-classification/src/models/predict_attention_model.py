import pandas as pd
import torchaudio
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from src.data.dementia_dataset import AudioDatasetExternal


def test(test_names, test_true, model, config, device):
    all_pred_results, all_probs_results = [], []
    true_info = {'audio_path': test_names, 'true_labels': test_true}

    model.to(device)

    for test_audio_p in test_names:
        pred_label, mean_pred = infer_one_audio(test_audio_p, config, model, device, True)

        all_pred_results.append(pred_label)
        all_probs_results.append(mean_pred)

    fpr, tpr, thresholds = roc_curve(test_true, all_probs_results)
    tn, fp, fn, tp = confusion_matrix(test_true, all_pred_results).ravel()
    specificity = tn / (tn + fp)

    print('Before choosing the best threshold')
    print(f'Specificity {specificity}')
    print(f'Accuracy {accuracy_score(all_pred_results, test_true)}')
    print(f'F1 score {f1_score(all_pred_results, test_true)}')

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print('After choosing the best threshold')
    all_pred_results = [1 if i > optimal_threshold else 0 for i in all_probs_results]
    tn, fp, fn, tp = confusion_matrix(test_true, all_pred_results).ravel()
    specificity = tn / (tn + fp)
    print(f'Specificity {specificity}')
    print(f'Accuracy {accuracy_score(all_pred_results, test_true)}')
    print(f'F1 score {f1_score(all_pred_results, test_true)}')

    true_info['predicted_labels'] = all_pred_results
    true_info['probabilities'] = all_probs_results
    pd.DataFrame.from_dict(true_info).to_csv('predicted_results.csv')


def infer_one_audio(audio_p, config, model, device, p2db=True):
    sigmoid = nn.Sigmoid()
    feature_extractor = config[f'{config["data_type"]}_extractor']
    frames = AudioDatasetExternal.split_audio_by_frames(audio_p, config)

    features = [feature_extractor(frame) for frame in frames if frame.shape[1] > 1]
    if p2db:
        p2b_transformer = torchaudio.transforms.AmplitudeToDB()
        features = [p2b_transformer(f) for f in features]
    all_results = []

    for feature in features:
      feature = feature.unsqueeze(0).to(device)
      result = sigmoid(model(feature))
      all_results.append(result.cpu())
    all_results = torch.stack(all_results)

    mean_r = torch.mean(all_results)
    if mean_r > 0.5:
      return 1, mean_r.item()
    return 0, mean_r.item()
