import torch

class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        super(AnomalyDataset, self).__init__
        self.dataset = df

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        row = self.dataset[idx]
        data = torch.from_numpy(row).float()
        return data

    

