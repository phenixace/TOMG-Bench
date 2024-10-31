from torch.utils.data import Dataset
import pandas as pd
import selfies

SYSTEM_HEAD = "You are working as an assistant of a chemist user. Please follow the instruction of the chemist and generate a molecule that satisfies the requirements of the chemist user. You could think step by step, but your final response should be a SMILES string. For example, 'Molecule: [SMILES STRING]'."
SYSTEM_HEAD_JSON = "You are working as an assistant of a chemist user. Please follow the instruction of the chemist and generate a molecule that satisfies the requirements of the chemist user. Your final response should be a JSON object with the key 'molecule' and the value as a SMILES string. For example, {\"molecule\": \"[SMILES_STRING]\"}."

class OMGDataset(Dataset):
    def __init__(self, maintask, subtask, json_check=False, use_selfies=False):
        filename = f'./data/benchmarks/open_generation/{maintask}/{subtask}/test.csv'
        self.data = pd.read_csv(filename)
        self.instruction = self.data["Instruction"].tolist()
        if use_selfies and maintask in ["MolEdit", "MolOpt"]:
            mol_selfies = [selfies.encoder(mol) for mol in self.data["molecule"].tolist()]
            for i in range(len(self.instruction)):
                self.instruction[i] = self.instruction[i].replace(self.data["molecule"][i], mol_selfies[i])
        self.json_check = json_check

    def __len__(self):
        return len(self.instruction)

    def __getitem__(self, idx):
        query = self.instruction[idx]
        if self.json_check:
            message = [
                {"role": "system", "content": SYSTEM_HEAD_JSON},
                {"role": "user", "content": query},
            ]
        else:
            message = [
                {"role": "system", "content": SYSTEM_HEAD},
                {"role": "user", "content": query},
            ]
        return message
    
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
    dataset = OMGDataset('MolEdit', 'AddComponent', json_check=True, use_selfies=True)
    print(len(dataset))
    print(dataset[0])
    print(dataset.instruction[0])