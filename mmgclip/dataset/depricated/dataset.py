from torch.utils.data.dataset import Dataset
from ..utils.data_utils import create_dataset_df, create_dataset_path
from ..utils.global_utils import seeding
from typing import List
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
from ..utils.logger import logger
from ..prompts.generator import generate_label_prompt_sentence, generate_label_prompt_report
from PIL import Image
import albumentations as A
from torchvision import transforms
from ..prompts.enums import * 
import numpy as np
from albumentations.pytorch import ToTensorV2

class ImageLabelDataset(Dataset):
    def __init__(
            self,
            config: object,
            data_folder = '0/02',
            split_type: str = 'train'
        ):
        super().__init__()

        self.config = config        
        self.split = split_type
        
        self.data_path = os.path.join(self.config.base.features_export_dir, data_folder)
        # self.transform = transforms.Compose([transforms.ToTensor()])

        self.search_col = self.config.dataset.config.search_col \
            if not (self.config.dataset.config.generate_label_prompt_sentence or self.config.dataset.config.generate_label_prompt_report)\
                 else "search_col" # random string when we generate prompt sentence as it is not needed
        
        self.new_col = self.search_col + '_new' # new column to create

        # load the annotated dataset to structure the labels and map them to the .pth loaded files
        self.dataset_df = create_dataset_df(config=self.config).sort_values('image_id').reset_index(drop=True)
        print(self.dataset_df.sample())

        # prepare the df dataset classification labels
        dataset_callback = generate_label_prompt_sentence if self.config.dataset.config.generate_label_prompt_sentence \
            else generate_label_prompt_report if self.config.dataset.config.generate_label_prompt_report \
            else None
        
        self.dataset_df = self._process_dataset_labels(
            self.dataset_df, 
            new_col=self.new_col, 
            search_col= self.search_col, 
            callback=dataset_callback)
        # print(self.dataset_df.sample())

        if self.config.dataset.config.generate_label_prompt_sentence: logger.info(f"Generating prompts for column {self.search_col} ..")

        # prepare the image features dataset
        # self.dataset_pth = create_dataset_path(path=self.data_path).sort_values('image_id').reset_index(drop=True)

        # merge the label column to the pth dataset
        # self.dataset_pth = pd.merge(self.dataset_pth, self.dataset_df[['image_id', 'image_label', 'mass_shape', 'mass_margin', 'has_mass', self.new_col]], on='image_id', how='inner') # on='image_id' how='left'
        self.dataset_pth = self.dataset_df[['image_id', 'image_path', 'image_label', 'mass_shape', 'mass_margin', 'has_mass', self.new_col]]
        # print(self.dataset_pth.sample())

        # exclude uncertain label
        self.dataset_pth = self.dataset_pth[self.dataset_pth['image_label'] != 2]
        logger.info(f"Total dataset length: {len(self.dataset_pth)}.") # Column {self.dataset_pth[self.processed_label].value_counts()}
        logger.info(f"Value Counts: \n {self.dataset_pth[self.new_col].value_counts()}")
        print(self.dataset_pth.sample())

        # setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.config.tokenizer.config.tokenizer_name)
        self.sequence_length = self.config.tokenizer.config.sequence_length

    def random_split(self, dataset, split_type):
        ratio = self.config.dataset.split.train_split_ratio if split_type == 'train' else self.config.dataset.split.test_split_ratio
        train_size = int(ratio * len(dataset))
        val_size = len(dataset) - train_size

        self.split_type = split_type
        
        train_split, val_split = \
            torch.utils.data.random_split(
                dataset, 
                [train_size, val_size], 
                generator = torch.Generator().manual_seed(self.config.base.seed))
        
        return train_split, val_split

    def _process_dataset_labels(self, dataset, new_col, search_col, callback = None):
        '''Process the dataset to categorize the dataframe rows into classification classes.
        
        Current implemented processing:
        1. Categorize based on mass_margin labels to `has_mass` column: label 0: no mass, label 1: hass one of the mass types

        Args:
            dataset (pd.DataFrame): the dataset to process

        Returns: 
            dataset (pd.DataFrame): the same dataset processed
        
        '''

        def _row_processor(val):
            '''
            Process the dataframe rows based on the dataset config.

            Args:
                val: a value from a column.

            Returns:
                label (str): depending on the dataset config conditions.

            '''

            if callback is not None:
                # callbacks to generate prompts
                if callback.__name__ == "generate_label_prompt_sentence":
                    # TODO: this must be generalized, it is based on benign val = 0, malig val = 1 for generating a single sentence for a label
                    label = callback(self.config.dataset.template.label[0], self.config.dataset.template.template_keys[0], n=1, template = self.config.dataset.template.prompt_template)[0] if val == 0 \
                        else callback(self.config.dataset.template.label[1], self.config.dataset.template.template_keys[1], n=1, template = self.config.dataset.template.prompt_template)[0]
                
            elif self.config.dataset.config.enums_class == 'BenignMalignantDatasetLabels':
                EnumClass = globals().get(self.config.dataset.config.enums_class)

                label = EnumClass(0).name if val == EnumClass(0).value else EnumClass(1).name

            elif self.config.dataset.config.enums_class == 'MassShapeLabels':
                EnumClass = globals().get(self.config.dataset.config.enums_class)

                label = str(val[0]) # (str) we take the first mass shape whether it is one or more, or if it has value -1 (undefined label)

                if label == "-1":
                    label = EnumClass(0).name
                elif label == 'lobular':
                    label = str(EnumClass.oval.name)

                label = label.lower()
            
            elif self.config.dataset.config.enums_class == 'MassMarginLabels':
                EnumClass = globals().get(self.config.dataset.config.enums_class)
                
                label = str(val[0]) # (str) we take the first mass margin whether it is one or more

                if label == "-1":
                    label = EnumClass(0).name
                elif label == "Ill defined":
                    label = "illdefined"

                label = label.lower()

            return label

        if callback is not None and callback.__name__ == "generate_label_prompt_report":
            # need to pass all the dataset, not just a specific column
            # the new column will be updated inside the function
            dataset = generate_label_prompt_report(dataset, new_col)
            return dataset
        
        dataset[new_col] = dataset[search_col].apply(_row_processor)

        return dataset

    def __len__(self):
        return len(self.dataset_pth)
    
    def _prepare_prompt_labels(self, index):
        '''Prepares all of the labels in the dataframe to be used for sentence training and label evaluation. Uses the same preprocessing logic.'''
        def _cast_labels(val, enums_class):
            if enums_class == 'MassShapeLabels':
                EnumClass = MassShapeLabels

                label = str(val[0]) # (str) we take the first mass shape whether it is one or more, or if it has value -1 (undefined label)

                if label == "-1":
                    label = EnumClass(0).name
                elif label == 'lobular':
                    label = str(EnumClass.oval.name)

                label = label.lower()
            
            elif enums_class == 'MassMarginLabels':
                EnumClass = MassMarginLabels
                
                label = str(val[0]) # (str) we take the first mass margin whether it is one or more

                if label == "-1":
                    label = EnumClass(0).name
                elif label == "Ill defined":
                    label = "illdefined"

                label = label.lower()

            return label

        has_mass = self.dataset_pth.iloc[index]['has_mass']
        mass_shape = self.dataset_pth.iloc[index]['mass_shape']
        mass_margin = self.dataset_pth.iloc[index]['mass_margin']
        image_label = self.dataset_pth.iloc[index]['image_label'] # 0 - benign, 1 - malignant

        labels = {"HasMassLabels": 1 if has_mass else 0,
                  "MassShapeLabels": _cast_labels(mass_shape, enums_class="MassShapeLabels"),
                  "MassMarginLabels": _cast_labels(mass_margin, enums_class="MassMarginLabels"),
                  "BenignMalignantDatasetLabels": BenignMalignantDatasetLabels(0).name if image_label == BenignMalignantDatasetLabels(0).value else BenignMalignantDatasetLabels(1).name
                  }
        
        return labels
    
    def _transform(self, image, split = None):
        # # Define transformations
        # train_transform = A.Compose([
        #     A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.1)),
        #     A.CLAHE(clip_limit=4.0),
        #     A.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
        #     A.CenterCrop(height=224, width=224),
        #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ToTensorV2(),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        tr = A.Compose([
            A.Resize(height=224, width=224),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # if split == 'train':
        #     tr = train_transform
        # elif split == 'test': # same as valid
        #     tr = valid_transform

        image = tr(image=np.array(image))['image']        
        return image
        
    def __getitem__(self, index):
        image_view = self.dataset_pth.iloc[index]

        # read and prepare the image
        image_features = Image.open(image_view['image_path']).convert("RGB")
        # image_features = torch.load(image_view['image_path'])
        if self.split is not None:
            image_features = self._transform(image=image_features, split=self.split)

        image_description = image_view[self.new_col]
        image_label = torch.tensor([image_view['image_label']])
        prompt_labels = self._prepare_prompt_labels(index)
        image_id = image_view['image_id']

        return {"image_features": image_features, "image_description": image_description, "image_label": image_label, "image_id": image_id, "prompt_labels": prompt_labels} 
    
    def collate_fn(self, instances: List):
        image_features = torch.stack([ins["image_features"] for ins in instances], dim=0)
        image_label = torch.stack([ins["image_label"] for ins in instances], dim=0)
        image_description = [ins["image_description"] for ins in instances]
        text_tokens = self.tokenizer(image_description, padding="max_length", truncation=True, return_tensors="pt", max_length=self.sequence_length)
        image_id = [ins["image_id"] for ins in instances]
        prompt_labels = [ins["prompt_labels"] for ins in instances]
        
        return {"image_features": image_features, "text_tokens": text_tokens, "image_description": image_description, "image_label": image_label, "image_id": image_id, "prompt_labels": prompt_labels}


# ImageTextDataset
# for report generation
