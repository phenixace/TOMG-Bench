from torch.utils.data import Dataset
import pandas as pd

class OMGDataset(Dataset):
    def __init__(self, maintask, subtask):
        filename = f'../data/benchmarks/open_generation/{maintask}/{subtask}/test.csv'
        self.data = pd.read_csv(filename)['Instruction'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data[idx]

        return query
    
class TMGDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target
    
class InsTDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target

class SourceDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target
    
if __name__ == '__main__':
    dataset = OMGDataset('OpenMol', 'AtomNum')
    print(len(dataset))
    print(dataset[0])