import torch
import librosa
from torch.utils.data import Dataset


class DementiaDatasetSpecs(Dataset):
    def __init__(self, info, transform=True):
        self.info = info
        self.use_transform = transform

    def __len__(self):
        return len(self.info)

    def get_label(self, label):
        if 'dementia' in label:
            return 1
        else:
            return 0

    def transform(self, spec, std=17.8581, mean=-57.4955):
        return (spec - mean) / std

    def __getitem__(self, idx):
        spec = self.info[idx]['spec']
        spec = torch.tensor(librosa.power_to_db(spec))

        if self.use_transform:
            spec = self.transform(spec)

        target = self.get_label(self.info[idx]['condition'])

        return {"spec": spec, "target": target}


def collate_batch(batch):
    labels, specs = [], []
    for batch_info in batch:
        if isinstance(batch_info['spec'], str):
            continue
        labels.append(batch_info['target'])
        specs.append(batch_info['spec'].unsqueeze(0) if batch_info['spec'].shape[0] != 1 else batch_info['spec'])

    labels = torch.tensor(labels, dtype=torch.int64)
    specs = torch.stack(specs, 0)

    return {"specs": specs, "labels": labels}
