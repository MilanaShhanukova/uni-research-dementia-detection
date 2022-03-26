import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, iterator, optimizer, criterion, scheduler=None, epoch=0, transformer=False, sam=True):
    softmax = nn.Softmax()
    model.train()

    epoch_loss, accuracy, f1, recall, precision, roc_auc = 0, 0, 0, 0, 0, 0
    num_auc = 0
    for i, batch in enumerate(iterator):
        specs = batch['specs'].to(device)
        batch_size = specs.shape[0]
        labels = batch['labels']

        optimizer.zero_grad()

        if transformer:
            specs_batch_shape = specs.shape
            specs = specs.squeeze().reshape(specs_batch_shape[0], specs_batch_shape[-1], specs_batch_shape[2])
            output = model(specs, task='ft_cls').flatten()
        else:
            output = model(specs)

        loss = criterion(output.cpu().type(torch.float), labels.type(torch.float))
        loss.backward()
        if scheduler is not None:
            scheduler.step()
        if sam:
            optimizer.first_step(zero_grad=True)

            output = model(specs)
            loss = criterion(output.cpu().type(torch.float), labels.type(torch.float))
            loss.backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        epoch_loss += loss.item()

        # calculate metrics
        result = [1 if i >= 0.5 else 0 for i in output]
        output_softmaxed = softmax(output)
        labels = labels.tolist()

        accuracy += accuracy_score(labels, result)
        f1 += f1_score(labels, result)
        recall += recall_score(labels, result)
        precision += precision_score(labels, result)
        try:
            roc_auc += roc_auc_score(labels, output_softmaxed.cpu().detach().numpy())
            num_auc += 1
        except:
            continue

    accuracy /= (i + 1)
    f1 /= (i + 1)
    epoch_loss /= (i + 1)
    recall /= (i + 1)
    precision /= (i + 1)
    roc_auc /= num_auc

    return epoch_loss, accuracy, f1, recall, precision, roc_auc


def evaluate(model, iterator, criterion, epoch, transformer=False):
    softmax = nn.Softmax()
    model.eval()
    epoch_loss, accuracy, f1, precision, recall, roc_auc = 0, 0, 0, 0, 0, 0
    num_auc = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            specs = batch['specs'].to(device)
            batch_size = specs.shape[0]

            labels = batch['labels']

            if transformer:
                specs_batch_shape = specs.shape
                specs = specs.squeeze().reshape(specs_batch_shape[0], specs_batch_shape[-1], specs_batch_shape[2])
                output = model(specs, task='ft_cls').flatten()
            else:
                output = model(specs)

            loss = criterion(output.cpu().type(torch.float), labels.type(torch.float))

            epoch_loss += loss.item()
            result = [1 if i >= 0.5 else 0 for i in output]
            labels = labels.tolist()

            output_softmaxed = softmax(output)

            accuracy += accuracy_score(labels, result)
            f1 += f1_score(labels, result)
            recall += recall_score(labels, result)
            precision += precision_score(labels, result)
            try:
                roc_auc += roc_auc_score(labels, output_softmaxed.cpu().detach().numpy())
                num_auc += 1
            except:
                continue

        accuracy /= (i + 1)
        f1 /= (i + 1)
        epoch_loss /= (i + 1)
        recall /= (i + 1)
        precision /= (i + 1)
        roc_auc /= num_auc + 1

        return epoch_loss, accuracy, f1, recall, precision, roc_auc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
