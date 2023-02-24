import pandas as pd
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    """
    Class that contains articles abstracts and titles
    """

    def __init__(self, csv_dir, transform=None, target_transform=None):
        """
        @param csv_dir: string, directory of csv file containing columns 'title' and 'abstract'
        @param transform: function, function to apply to abstracts
        @param target_transform: function, function to apply to titles
        """
        df = pd.read_csv(csv_dir, usecols=['abstract', 'title'])
        self.x = list(df['abstract'])
        self.y = list(df['title'])
        df = None  # free memory
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        abstract, title = self.x[idx], self.y[idx]
        if self.transform:
            abstract = self.transform(abstract)
        if self.target_transform:
            title = self.target_transform(title)
        return abstract, title

    def __len__(self):
        return len(self.x)
