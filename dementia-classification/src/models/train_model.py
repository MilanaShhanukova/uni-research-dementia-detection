import torch
import torch.nn as nn
import pandas as pd
import torchaudio
from src.dementia_dataset import AudioDatasetExternal
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import warnings
warnings.filterwarnings("ignore")

def get_weight(data_path):
  data = pd.read_csv(data_path)
  values = data['label'].value_counts()

  return values['Control'] / values['Dementia']

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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, criterion, optimizer, scheduler, device='cuda', transformer=False, threshold=0.5):
    model.train()

    epoch_loss, accuracy, f1, recall, precision = 0, 0, 0, 0, 0

    for i, batch in enumerate(iterator):
        specs = batch['features'].to(device)
        labels = batch['label']

        optimizer.zero_grad()

        if transformer:
            specs_batch_shape = specs.shape
            specs = specs.squeeze().reshape(specs_batch_shape[0], specs_batch_shape[-1], specs_batch_shape[2])
            output = model(specs, task='ft_cls').flatten()
        else:
            output = model(specs)

        loss = criterion(output.cpu().type(torch.float), labels.type(torch.float))
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()

        #calculate metrics
        result = [1 if i >= threshold else 0 for i in output]
        labels = labels.tolist()

        accuracy += accuracy_score(labels, result)
        f1 += f1_score(labels, result)
        recall += recall_score(labels, result)
        precision += precision_score(labels, result)


    wandb.log({"loss_train": epoch_loss / (i+1), "accuracy_train": accuracy / (i+1), "f1_train": f1 / (i+1),
                           "recall_train": recall / (i+1), "precision_train": precision / (i+1)})

    accuracy /= (i+1)
    f1 /= (i+1)
    epoch_loss /= (i+1)
    recall /= (i+1)
    precision /= (i+1)

    return epoch_loss, accuracy, f1, recall, precision

def evaluate(model, iterator, criterion, device='cuda', transformer=False, threshold=0.5):
    """
    Evaluate model during training.
    """
    model.eval()

    epoch_loss, accuracy, f1, precision, recall = 0, 0, 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            specs = batch['features'].to(device)
            labels = batch['label']

            if transformer:
                specs_batch_shape = specs.shape
                specs = specs.squeeze().reshape(specs_batch_shape[0], specs_batch_shape[-1], specs_batch_shape[2])
                output = model(specs, task='ft_cls').flatten()
            else:
                output = model(specs)

            loss = criterion(output.cpu().type(torch.float), labels.type(torch.float))

            epoch_loss += loss.item()
            result = [1 if i >= threshold else 0 for i in output]
            labels = labels.tolist()

            accuracy += accuracy_score(labels, result)
            f1 += f1_score(labels, result)
            recall += recall_score(labels, result)
            precision += precision_score(labels, result)

    wandb.log({"loss_val": epoch_loss / (i+1), "accuracy_val": accuracy / (i+1), "f1_val": f1 / (i+1), "recall_val": recall / (i+1), "precision_val": precision / (i+1)})

    accuracy /= (i+1)
    f1 /= (i+1)
    epoch_loss /= (i+1)
    recall /= (i+1)
    precision /= (i+1)

    return epoch_loss, accuracy, f1,  recall, precision
