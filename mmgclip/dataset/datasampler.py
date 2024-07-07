import torch
import pandas as pd
import numpy as np
from ..utils.logger import logger

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        dataset: dataset object to sample from.
        class_name: the label or target class for sampling.
    """

    def __init__(
        self,
        dataset,
        class_name: str = None 
    ):
        logger.info("Using a sampler for handling class imbalance.")

        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        # the class name that is used as the label
        self.class_name = class_name

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        self.class_indices = [np.where(df["label"].values == label)[0] for label in label_to_count.index]

    def _get_labels(self, dataset):
        return [label[self.class_name] for _, label in enumerate(dataset)]

    def __iter__(self):
        # Calculate the number of samples per batch
        samples_per_batch = self.num_samples // len(self.class_indices)

        # Iterate over batches
        for _ in range(samples_per_batch):
            # Sampling with replacement from each class in a balanced way
            class_indices_iterators = [iter(np.random.choice(indices, samples_per_batch, replace=True)) for indices in self.class_indices]
            for iterator in class_indices_iterators:
                yield next(iterator)

    def __len__(self):
        return self.num_samples