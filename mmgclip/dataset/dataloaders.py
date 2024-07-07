import torch.utils.data as data 
from .datasampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader, Subset
import torch

class DataLoaders:
    def __init__(
            self,
            config : object,
            dataset_split: data.Dataset,
        ):
        super().__init__()

        self.config = config
        self.dataset_split = dataset_split

    def get_dataloader(self, shuffle=True, batch_size = 32, drop_last = False, pin_memory=True, collate_fn = None, num_workers=8, prefetch_factor=16, label_class_name = 'image_description', use_sampler = False):

        if use_sampler:
            return data.DataLoader(
                self.dataset_split,
                sampler= ImbalancedDatasetSampler(self.dataset_split, class_name=label_class_name),
                batch_size = batch_size,
                drop_last = drop_last,
                pin_memory = pin_memory,
                num_workers = num_workers,
                collate_fn = collate_fn,
                prefetch_factor=prefetch_factor
            )
        
        return data.DataLoader(
            self.dataset_split,
            shuffle = shuffle,
            batch_size = batch_size,
            drop_last = drop_last,
            pin_memory = pin_memory,
            num_workers = num_workers,
            collate_fn = collate_fn,
            prefetch_factor=prefetch_factor
        )
    
def dataloader_percentage(dataloader, config, collate_fn):
    # Calculate the number of samples to keep
    num_samples_to_keep = int(len(dataloader.dataset) * config.dataset.percentage.config.percentage)

    # Generate random indices for selecting samples
    indices = torch.randperm(len(dataloader.dataset)).tolist()

    dlconf =  config.dataloader.train
    dlconf.pop('use_sampler', None)     # remove use_sampler as we will use a specific sampler
    dlconf.pop('shuffle', None)

    return data.DataLoader(dataloader.dataset,
                                sampler=torch.utils.data.SubsetRandomSampler(indices[:num_samples_to_keep]),
                                **dlconf,
                                collate_fn = collate_fn
                                )