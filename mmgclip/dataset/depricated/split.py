import torch
from torch.utils.data.dataset import Dataset

class SplitDataset:
    def __init__(
            self,
            dataset: Dataset,
            config: object,
            type: str = 'train'
            ):
        
        super().__init__()

        self.dataset = dataset
        self.config = config
        self.type = type
        
    def random_split(self):
        ratio = self.config.dataset.split.train_split_ratio if self.type == 'train' else self.config.dataset.split.test_split_ratio
        train_size = int(ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_split, val_split = \
            torch.utils.data.random_split(
                self.dataset, 
                [train_size, val_size], 
                generator = torch.Generator().manual_seed(self.config.base.seed))
        
        return train_split, val_split

