from torch.utils.data.dataset import Dataset
from ..utils.data_utils import create_dataset_df, create_dataset_path, process_class_list
from typing import List
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
from ..utils.logger import logger
from ..prompts.generator import *
from ..prompts.enums import * 
import ast

class ImageLabelDataset(Dataset):
    def __init__(
            self,
            config: object,
            data_folder = '0/02',
            split = None
        ):
        super().__init__()

        self.config = config

        self.data_path = os.path.join(self.config.base.features_export_dir, data_folder)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.split = split
        
        self.search_col = self.config.dataset.config.search_col \
            if not (self.config.dataset.config.generate_label_prompt_sentence or self.config.dataset.config.generate_label_prompt_report)\
                 else "search_col" # random string when we generate prompt sentence as it is not needed
        
        self.new_col = self.search_col + '_new' # new column to create

        # load the annotated dataset to structure the labels and map them to the .pth loaded files
        self.dataset_df = create_dataset_df(config=self.config).sort_values('image_id').reset_index(drop=True)

        # prepare the df dataset classification labels
        dataset_callback = generate_label_prompt_sentence if self.config.dataset.config.generate_label_prompt_sentence \
            else generate_label_prompt_report if self.config.dataset.config.generate_label_prompt_report \
            else None
                
        self.dataset_df = self._process_dataset_labels(
            self.dataset_df, 
            new_col=self.new_col, 
            search_col= self.search_col, 
            callback=dataset_callback)

        if self.config.dataset.config.generate_label_prompt_sentence: logger.info(f"Generating prompts for column {self.search_col} ..")

        # prepare the image features dataset
        self.dataset_pth = create_dataset_path(path=self.data_path).sort_values('image_id').reset_index(drop=True)

        # merge the label column to the pth dataset
        self.dataset_pth = pd.merge(self.dataset_pth, self.dataset_df[['image_id', 'image_label', 'mass_shape', 'mass_margin', 'has_mass', 'has_architectural_distortion', 'has_calc', self.new_col]], on='image_id', how='inner') # on='image_id' how='left'
        # print(self.dataset_pth.sample())

        # exclude uncertain label
        self.dataset_pth = self.dataset_pth[self.dataset_pth['image_label'] != 2]
        logger.info(f"Total dataset length: {len(self.dataset_pth)}.") # Column {self.dataset_pth[self.processed_label].value_counts()}
        logger.info(f"Value Counts: \n {self.dataset_pth[self.new_col].value_counts()}")
        logger.info(self.dataset_pth.sample())
        # print(self.dataset_pth.sample())

        # remove empty rows ? to experiment
        # self.dataset_pth = self.dataset_pth[self.dataset_pth[self.new_col] != ""]

        # export
        self.dataset_pth[self.new_col].to_csv(os.path.join(self.config.base.export_dir, 'image_description.txt'), index=False, header=False, sep=' ', mode='a')

        # setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.config.tokenizer.config.tokenizer_name)
        self.sequence_length = self.config.tokenizer.config.sequence_length

    def random_split(self, dataset, split):
        ratio = self.config.dataset.split.train_split_ratio if split == 'train' else self.config.dataset.split.test_split_ratio
        train_size = int(ratio * len(dataset))
        val_size = len(dataset) - train_size

        self.split = split
        
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

                # elif label == "Ill defined":
                #     label = "illdefined"

                label = label.lower()

            elif self.config.dataset.config.enums_class == "HasMassLabels":
                EnumClass = globals().get(self.config.dataset.config.enums_class)

                label = 1 if val else 0 # 1 if has_mass else 0,
                label = get_key_from_value(EnumClass, label)

                if label == "nomass":
                    label = "no mass"

                label = label.lower()

            elif self.config.dataset.config.enums_class == "HasArchDistortion":
                EnumClass = globals().get(self.config.dataset.config.enums_class)

                label = 1 if val else 0 # 1 if has_architectural_distortion else 0,
                label = get_key_from_value(EnumClass, label)

                if label == "noarchitecturaldistortion":
                    label = "no architectural distortion"
                elif label == "displayedarchitecturaldistortion":
                    label = "displayed architectural distortion"

                label = label.lower()

            elif self.config.dataset.config.enums_class == "HasCalcification":
                EnumClass = globals().get(self.config.dataset.config.enums_class)
                
                label = 1 if val else 0 # 1 if has_calc else 0,
                label = get_key_from_value(EnumClass, label)

                if label == "noncalcified":
                    label = "non-calcified"
                elif label == "hascalcification":
                    label = "has calcification"

                label = label.lower()

            return label

        if callback is not None and callback.__name__ == "generate_label_prompt_report":
            # need to pass all the dataset, not just a specific column
            # the new column will be updated inside the function
            # dataset = generate_label_prompt_report(dataset, new_col)

            def generate_study_gtr_report(row):

                ''' Updated prompts to follow exam prompts.

                    image_id                                                            p0200000202cl
                    image_label                                                                     0
                    mass_margin                                 [Ill defined, -1, -1, -1, -1, -1, -1]
                    mass_shape                                         [Oval, -1, -1, -1, -1, -1, -1]
                    has_mass                                                                     True
                    has_architectural_distortion                                                False
                    has_calc                                                                    False
                    image_path                      /storage/Features/features/png_archive/2D_100m...
                '''
                prompts = []

                # check for mass only
                if row['has_mass']:
                    malign = 'malignant' if row['image_label'] == 1 else 'benign'
                    mass_margin = list(set([val.lower() for val in row['mass_margin'] if val != -1]))
                    mass_margin = "unknown" if len(mass_margin) == 0 else mass_margin[0]

                    mass_shape = list(set([val.lower() for val in row['mass_shape'] if val != -1]))
                    mass_shape = "unknown" if len(mass_shape) == 0 else mass_shape[0]

                    mass_prompt = generate_gtr_prompt_sentence(key='gtr_mass:True', n=1, M_MALIG=malign, M_MARG=mass_margin, M_SHAPE=mass_shape)
                    prompts.append(mass_prompt)

                # check for calc only
                if row['has_calc']:
                    malign = 'malignant' if row['image_label'] == 1 else 'benign'

                    calc_prompt = generate_gtr_prompt_sentence(key='gtr_calc:True', n=1, C_MALIG=malign)
                    prompts.append(calc_prompt)

                # check for arch distortion
                if row["has_architectural_distortion"]:
                    prompts.append(generate_gtr_prompt_sentence(key=f'gtr_is_architectural_distortion:{row["has_architectural_distortion"]}', n=1))

                return ' '.join(prompts)

            dataset[new_col] = dataset.apply(generate_study_gtr_report, axis=1)
            
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
                    
                # elif label == "Ill defined":
                #     label = "illdefined"

                label = label.lower()

            elif enums_class == "HasMassLabels":
                EnumClass = HasMassLabels

                label = 1 if val else 0 # 1 if has_mass else 0,
                label = get_key_from_value(EnumClass, label)

                if label == "nomass":
                    label = "no mass"

                label = label.lower()

            elif enums_class == "HasArchDistortion":
                EnumClass = HasArchDistortion
                label = 1 if val else 0 # 1 if has_architectural_distortion else 0,
                label = get_key_from_value(EnumClass, label)

                if label == "noarchitecturaldistortion":
                    label = "no architectural distortion"
                elif label == "displayedarchitecturaldistortion":
                    label = "displayed architectural distortion"

                label = label.lower()

            elif enums_class == "HasCalcification":
                EnumClass = HasCalcification
                label = 1 if val else 0 # 1 if has_calc else 0,
                label = get_key_from_value(EnumClass, label)

                if label == "noncalcified":
                    label = "non-calcified"
                elif label == "hascalcification":
                    label = "has calcification"

                label = label.lower()

            return label

        has_mass = self.dataset_pth.iloc[index]['has_mass']
        has_calc = self.dataset_pth.iloc[index]['has_calc']
        has_architectural_distortion = self.dataset_pth.iloc[index]['has_architectural_distortion']
        mass_shape = self.dataset_pth.iloc[index]['mass_shape']
        mass_margin = self.dataset_pth.iloc[index]['mass_margin']
        image_label = self.dataset_pth.iloc[index]['image_label'] # 0 - benign, 1 - malignant

        labels = {"HasMassLabels": _cast_labels(has_mass, enums_class="HasMassLabels"),
                  "MassShapeLabels": _cast_labels(mass_shape, enums_class="MassShapeLabels"),
                  "MassMarginLabels": _cast_labels(mass_margin, enums_class="MassMarginLabels"),
                  "BenignMalignantDatasetLabels": BenignMalignantDatasetLabels(0).name if image_label == BenignMalignantDatasetLabels(0).value else BenignMalignantDatasetLabels(1).name,
                  "HasArchDistortion": _cast_labels(has_architectural_distortion, enums_class="HasArchDistortion"),
                  "HasCalcification": _cast_labels(has_calc, enums_class="HasCalcification"),
                  }
        
        return labels
        
    def __getitem__(self, index):
        image_view = self.dataset_pth.iloc[index]        
        image_description = image_view[self.new_col]
        image_features = torch.load(image_view['image_path'])
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
    
class StudyReportDataset(Dataset):
    def __init__(
            self,
            config: object,
            # data_folder = '0/02',
            split = None
        ):
        super().__init__()

        self.config = config
        self.split = split

        '''
        Steps to obtain the file for `final_reports_dataset`:
            1. Preprocess
            2. Translate
            3. Post-process
            4. Extract features
            5. Map study paths to extracted features paths
        '''
        self.final_reports_dataset = pd.read_csv(self.config.dataset.config.final_reports_dataset_path, encoding="unicode_escape", index_col=0, dtype = str)
        logger.info(f"Total dataset length: {len(self.final_reports_dataset)}.") # Column {self.dataset_pth[self.processed_label].value_counts()}
        logger.info(f"Loading image dataset from: {self.config.base.features_export_dir} folder...")
        logger.info(f"Loading report dataset from: {self.config.dataset.config.post_translation_fileid} folder...")
        
        # ----------------------------------- Prompt Generation using GTR Labels ------------------------------
        # -----------------------------------------------------------------------------------------------------
        if self.config.dataset.config.gtr_prompt_generation:
            logger.info("Generating prompts within the reports...")
            def generate_study_gtr_report(row):
                report = ""
                prompts = []

                if row.has_gtr_label:
                    # NOTE: any -1 label means not annotated
                    # NOTE: gtr file was read as str, validate correctly
                    row.labels = ast.literal_eval(row.labels)

                    # get the first sample for the patient id, the gtr file is at the view level, not patient level 
                    gtr_sample = self.final_reports_dataset_gtr[self.final_reports_dataset_gtr['full_study_id'] == f"{row['patient_id']}{row['study_id'][2:]}"].iloc[0]
                    
                    # check for mass only
                    if str(gtr_sample.gtr_mass) == "True": # and str(gtr_sample.gtr_calc) == "False":
                        # labels values
                        malign = 'malignant' if gtr_sample.gtr_malign == 'True' else 'benign'
                        mass_margin = get_key_from_value(gtr_MassMargin, int(gtr_sample.gtr_mass_margin))
                        
                        mass_margin = process_class_list([mass_margin])[0]

                        # generate one prompt sentence
                        mass_prompt = generate_gtr_prompt_sentence(key='gtr_mass:True', n=1, M_MALIG=malign, M_MARG=mass_margin, M_SHAPE=row.labels['masses']['shapes'])

                        # add birads score if it exists, as a single sentence.
                        if row.labels['birads'].lower() != "unknown":
                            # remove the ending '.', add a prompt of birads
                            mass_prompt = mass_prompt[:-1] + ", " + generate_gtr_prompt_sentence(key="row.labels['birads']:True", n=1, B_SCORE=row.labels['birads'])

                        prompts.append(mass_prompt)

                    # check for calc only
                    if str(gtr_sample.gtr_calc) == "True": # and str(gtr_sample.gtr_mass) == "False":
                        # TODO: add calc properties from reports labels dict, same for missing mass informations
                        malign = 'malignant' if gtr_sample.gtr_calc == 'True' else 'benign'

                        calc_prompt = generate_gtr_prompt_sentence(key='gtr_calc:True', n=1, C_MALIG=malign, C_DIST=row.labels['calcifications']['distribution'])
                        
                        # add birads score if it exists, as a single sentence.
                        if row.labels['birads'].lower() != "unknown":
                            # remove the ending '.', add a prompt of birads
                            calc_prompt = calc_prompt[:-1] + ", " + generate_gtr_prompt_sentence(key="row.labels['birads']:True", n=1, B_SCORE=row.labels['birads'])

                        prompts.append(calc_prompt)
                                            
                    # # check for both mass and calc
                    # if str(gtr_sample.gtr_mass) == "True" and str(gtr_sample.gtr_calc) == "True":
                    #     malign = 'malignant' if gtr_sample.gtr_malign == 'True' else 'benign'
                    #     mass_margin = get_key_from_value(gtr_MassMargin, int(gtr_sample.gtr_mass_margin))

                    #     # generate one prompt sentence
                    #     mass_prompt = generate_gtr_prompt_sentence(key='gtr_mass:True&gtr_calc:True', n=1, M_MALIG=malign, M_MARG=mass_margin, M_SHAPE=row.labels['masses']['shapes'])

                    #     # add birads score if it exists, as a single sentence.
                    #     if row.labels['birads'].lower() != "unknown":
                    #         # remove the ending '.', add a prompt of birads
                    #         mass_prompt = mass_prompt[:-1] + ", " + generate_gtr_prompt_sentence(key="row.labels['birads']:True", n=1, B_SCORE=row.labels['birads'])

                    #     prompts.append(mass_prompt)

                    # # both histology and biopsy information are reported in the image_impression column (use gtr back)
                    # # we use values > -1 if we take them from the gtr file; note that the impression column (histology) is also based on translation, and might contain benign/malignancy information
                    # histology_biopsy_prompt = row.image_impression + "."
                    # prompts.append(histology_biopsy_prompt)
                    # if int(gtr_sample.gtr_histology) > 0:
                    #     # get the histology based on the label
                    #     hist = gtr_Histology.get(int(gtr_sample.gtr_histology), "unknown")
                        
                    #     if hist != "unknown":
                    #         # generate one prompt sentence
                    #         hist_prompt = generate_gtr_prompt_sentence(key='gtr_histology>0', n=1, HISTOLOGY=hist)
                    #         prompts.append(hist_prompt)

                    # check for arch distortion
                    if str(gtr_sample.gtr_is_architectural_distortion) == True:
                        prompts.append(generate_gtr_prompt_sentence(key=f'gtr_is_architectural_distortion:{gtr_sample.gtr_is_architectural_distortion}', n=1))
                                            
                    # subtlety (refer to enums.cpp file)


                else:
                    # generate one prompt sentence
                    no_findings = generate_gtr_prompt_sentence(key='no_gtr', n=1)

                    # # add birads score 1
                    # # remove the ending '.', add a prompt of birads
                    # no_findings = no_findings[:-1] + ", " + generate_gtr_prompt_sentence(key="row.labels['birads']:True", n=1, B_SCORE="1")

                    prompts.append(no_findings)

                report = ' '.join(prompts)

                return report

            # labels used to generate prompts 
            self.final_reports_dataset_gtr = pd.read_csv(self.config.dataset.config.gt_path, dtype=str)
            
            # add patient id column for easier indexing
            self.final_reports_dataset_gtr['full_study_id'] = self.final_reports_dataset_gtr['gtr_path'].apply(lambda x: x.split('/')[-1][:10])

            # add has_gtr_label column in the `final_reports_dataset` if its patient id is in the list of gtr patient ids
            # self.final_reports_dataset['has_gtr_label'] = self.final_reports_dataset['full_study_id'].apply(lambda x: x in self.final_reports_dataset_gtr['patient_id'].values)
            self.final_reports_dataset['has_gtr_label'] = self.final_reports_dataset.apply(lambda x: f"{x['patient_id']}{x['study_id'][2:]}" in self.final_reports_dataset_gtr['full_study_id'].values, axis=1)

            # add prompt generated labels into a new prompt_generated column
            self.final_reports_dataset['prompt_generated'] = self.final_reports_dataset.apply(generate_study_gtr_report, axis=1)

            if self.config.dataset.config.use_gtr_prompts_only and self.config.dataset.config.gtr_prompt_generation:
                # assign the rows of image description the prompt values
                self.final_reports_dataset['image_description'] = self.final_reports_dataset['prompt_generated']

                # not all rows had gtr, eliminate empty rows
                self.final_reports_dataset = self.final_reports_dataset[self.final_reports_dataset['image_description'] != ""]
            else:
                # concatenate the generated prompts with the image_description
                self.final_reports_dataset['image_description'] = self.final_reports_dataset.apply(lambda x: x.prompt_generated + " " + x.image_description , axis=1)            
        # -----------------------------------------------------------------------------------------------------
        # ----------------------------------- Prompt Generation using GTR Labels ------------------------------

        logger.info(self.final_reports_dataset.sample())
        self.final_reports_dataset['image_description'].to_csv(os.path.join(self.config.base.export_dir, 'image_description.txt'), index=False, header=False, sep=' ', mode='a')

        # setup the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.config.tokenizer.config.tokenizer_name)
        self.sequence_length = self.config.tokenizer.config.sequence_length

    def random_split(self, dataset, split):
        ratio = self.config.dataset.split.train_split_ratio if split == 'train' else self.config.dataset.split.test_split_ratio
        train_size = int(ratio * len(dataset))
        val_size = len(dataset) - train_size

        self.split = split
        
        train_split, val_split = \
            torch.utils.data.random_split(
                dataset, 
                [train_size, val_size], 
                generator = torch.Generator().manual_seed(self.config.base.seed))
        
        return train_split, val_split

    def __len__(self):
        return len(self.final_reports_dataset)
    
    def _prepare_prompt_labels(self, index):
        exam_malig = int(self.final_reports_dataset.iloc[index]['is_malig']) # 0 - benign, 1 - malignant

        exam_labels = ast.literal_eval(self.final_reports_dataset.iloc[index]['labels'])
        exam_labels['BenignMalignantDatasetLabels'] = exam_malig # return 1 for malignancy directly
        exam_labels['MassShapeLabels'] = MassShapeLabels[exam_labels['masses']['shapes']].value
        exam_labels['BIRADS'] = exam_labels['birads']

        return exam_labels
    
    def __getitem__(self, index):
        study_row = self.final_reports_dataset.iloc[index]

        image_description = study_row['image_description']          # pre-/post-processed report in english (translated)
        image_impression = study_row['image_impression']            # impression section in english

        study_features = torch.load(study_row['study_path'])        # will act as the image_features
        study_label = torch.tensor([int(study_row['is_malig'])])    # will act as the image_label
        prompt_labels = self._prepare_prompt_labels(index)          # will act as the prompt_labels
        patient_id = study_row['patient_id']                        # will act as the patient_id

        return {"image_features": study_features, "image_description": image_description, "image_impression": image_impression, "image_label": study_label, "image_id": patient_id, "prompt_labels": prompt_labels} 

    def collate_fn(self, instances: List):
        image_features = torch.stack([ins["image_features"] for ins in instances], dim=0)
        image_label = torch.stack([ins["image_label"] for ins in instances], dim=0)
        
        image_description = [ins["image_description"] for ins in instances]         # en report
        image_impression = [ins["image_impression"] for ins in instances]           # en impression

        text_tokens = self.tokenizer(image_description, padding="max_length", truncation=True, return_tensors="pt", max_length=self.sequence_length)
        image_impression_tokens = self.tokenizer(image_impression, padding="max_length", truncation=True, return_tensors="pt", max_length=self.sequence_length)
        
        image_id = [ins["image_id"] for ins in instances]
        prompt_labels = [ins["prompt_labels"] for ins in instances]
        
        return {"image_features": image_features, "text_tokens": text_tokens, "image_impression_tokens":image_impression_tokens, "image_description": image_description, "image_label": image_label, "image_id": image_id, "prompt_labels": prompt_labels}

def get_dataset(dataset_name):
    """
    Returns the specified dataset controller class based on the provided dataset_name.

    Current available loss classes:
    1. 'ImageLabelDataset': An image-label dataset class, for both label and prompt training and classification.
    2. 'StudyReportDataset': A study-report dataset class, for study-report training and classification.

    Parameters:
    - dataset_name (str): The name of the dataset class. This class must be imported inside this file.

    Returns:
    - class: The corresponding dataset class.

    Raises:
    - ValueError: If the specified network_name does not correspond to a valid class.
    """
    network_class = globals().get(dataset_name, None)
    logger.info(f"Using {dataset_name} dataset.")
    
    if network_class is None:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")
    return network_class