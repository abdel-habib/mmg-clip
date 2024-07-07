from ..utils.logger import logger
from ..utils.global_utils import create_directory_if_not_exists
import pandas as pd
import os
import re
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageFeatureExtractor():
    def __init__(self, config= None, dataset = None):
        super().__init__()

        assert config is not None, 'Error in initializing the feature extractor. Missing training config object.'
        self.config = config

        # validate the dataset
        self.dataset = self._validate_dataset(dataset)

        # load the image encoder and set to eval mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_encoder  = torch.jit.load(self.config.networks.image_encoder.convnext_tiny_clf_path)
        self.image_encoder = self.image_encoder.to(self.device)
        self.image_encoder.eval()
        
        # Data utils
        self.transform = transforms.Compose([transforms.ToTensor()])

        # check if export directory exists, if not create one
        self.export_dir = create_directory_if_not_exists(self.config.base.features_export_dir)

    def _get_img_id(self, path):
        match =  re.search(r'p(\w+)', path)
        
        if match:
            return 'p'+match.group(1)
        
    def _validate_image_type(self, filepath, filetype = '.png'):
        return filepath.lower().endswith(filetype)

    def _validate_dataset(self, dataset):
        '''
        An internal validation function for the `dataset` passed.

        Args:
            dataset: passed argument in the class initialization.

        Returns:
            The same dataset, that is considered valid for image feature extration.  
        '''
        if isinstance(dataset, pd.DataFrame):
            if not 'file_name' and 'image_path' not in dataset.columns:
                raise ValueError("Error in the `dataset` dataframe passed. The dataframe doesn't contain the following columns `file_name` and `image_path`.")

        elif isinstance(dataset, str):
            raise NotImplemented('Handling a string directory dataset is not yet implemented.')
            # if not os.path.isdir(dataset):
            #     raise ValueError("Dataset folder doesn't exist.")
            
            # logger.warning("Passing a path will process all images found in all inner folders/directories.")
            # logger.warning(f"Preparing dataset {dataset}...")
            # _dataset = []

            # # Prepare the dataset to make sure that it follows the same structure as the dataframe version.
            # for path, subdirs, files in tqdm(os.walk(dataset)):
            #     for name in files:
            #         filepath = os.path.join(path, name)
            #         if self._validate_image_type(filepath, '.png'):
            #             _dataset.append([self._get_img_id(filepath), filepath])
            #         break
            #     print(_dataset)
                # break
        else:
            raise ValueError("Missing value for `dataset`. Please pass a valid Path or a dataset dataframe.")

        return dataset

    def extract(self):
        logger.info(f"Extracting and exporting features into {self.export_dir} directory.")

        torch.cuda.empty_cache()

        if isinstance(self.dataset, pd.DataFrame):
            with torch.no_grad():
                # for row in self.dataset
                for index, row in tqdm(self.dataset.iterrows()):
                    img_dirname = os.path.dirname(row['image_path'].split('2D_100micron/')[-1])
                    
                    try:
                        image_raw = Image.open(row['image_path'])
                        image_tensor = self.transform(image_raw).to(self.device)

                        # Dicom images in 16bits, while pngs are 8 bits to save space
                        image_tensor = 65535 * image_tensor
                        image_tensor = image_tensor.unsqueeze(0)

                        # Apply Forward pass in stages
                        image_tensor_norm = (image_tensor - 32767.5) / 32767.5
                        feature_map = self.image_encoder.features((image_tensor_norm))
                        features = self.image_encoder.avgpool(feature_map)
                        
                        # export the features as .pt files
                        # get the folder/file for export for this sample
                        # path should be /0/{patient_id[1:3]}/{patient_id}/st{study_id}/
                        # the path shouldn't hve the filename, we will save .pth file with that filename
                        sample_export_path = os.path.join(self.export_dir, img_dirname)
                        sample_export_pth_filename = os.path.join(self.export_dir, row['image_path'].split('2D_100micron/')[-1]).replace('.png', '.pth')

                        # create the directory
                        create_directory_if_not_exists(sample_export_path)

                        # detach the features from cuda device to allow pin_memory=True when other objects are not on cuda and avoid torch multiprocessing 'spawn'
                        features = features.detach().cpu()

                        # export the features as a .pth file
                        torch.save(features, sample_export_pth_filename)                

                    except Exception as e:
                        failed_txt_filepath = os.path.join(self.export_dir, 'failed.txt')
                        with open(failed_txt_filepath, "a") as myfile:
                            myfile.write(row['image_path'] + '\n' + str(e) + '\n\n')

class StudyFeatureExtractor():
    def __init__(self, config= None, dataset = None):
        super().__init__()

        assert config is not None, 'Error in initializing the feature extractor. Missing training config object.'
        self.config = config

        # validate the dataset
        self.dataset = self._validate_dataset(dataset)
        self.export_dir = os.path.join(self.config.base.features_export_dir)

        # load the image encoder and set to eval mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_encoder  = torch.jit.load(self.config.networks.image_encoder.convnext_tiny_clf_path)
        self.image_encoder = self.image_encoder.to(self.device)
        self.image_encoder.eval()
        
        # Data utils
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _validate_dataset(self, dataset):
        '''
        An internal validation function for the `dataset` passed.

        Args:
            dataset: passed argument in the class initialization.

        Returns:
            The same dataset, that is considered valid for image feature extration.  
        '''
        if isinstance(dataset, pd.DataFrame):
            if 'study_path' not in dataset.columns:
                raise ValueError("Error in the `dataset` dataframe passed. The dataframe doesn't contain the following column `study_path`.")

        elif isinstance(dataset, str):
            raise NotImplemented('Handling a string directory dataset is not yet implemented.')
            # if not os.path.isdir(dataset):
            #     raise ValueError("Dataset folder doesn't exist.")
            
            # logger.warning("Passing a path will process all images found in all inner folders/directories.")
            # logger.warning(f"Preparing dataset {dataset}...")
            # _dataset = []

            # # Prepare the dataset to make sure that it follows the same structure as the dataframe version.
            # for path, subdirs, files in tqdm(os.walk(dataset)):
            #     for name in files:
            #         filepath = os.path.join(path, name)
            #         if self._validate_image_type(filepath, '.png'):
            #             _dataset.append([self._get_img_id(filepath), filepath])
            #         break
            #     print(_dataset)
                # break
        else:
            raise ValueError("Missing value for `dataset`. Please pass a valid Path or a dataset dataframe.")

        return dataset
    
    def _get_patient_id(self, path):
        match =  re.search(r'\d{8}', path)

        if match:
            return match.group()

    def extract(self):
        logger.info(f"Extracting and exporting features into {self.export_dir} directory.")
        logger.info(f"Concatenating {self.config.dataset.config.n_images_per_study} images using {self.config.dataset.config.concatenate_features_method} method.")

        torch.cuda.empty_cache()

        with torch.no_grad():
            # for row in self.dataset, each row represent a study directory
            for index, row in tqdm(self.dataset.iterrows()):
                study_path = row['study_path']

                try:
                    # will hold all features extracted from each view inside a study
                    study_views_feature_vector = []

                    # iterate over study views
                    for idx, study_view_filename in enumerate(os.listdir(study_path)):
                        # to control how many images per study
                        if idx == self.config.dataset.config.n_images_per_study: break

                        # create a path for the image view
                        study_view_filepath = os.path.join(study_path, study_view_filename) 
                        
                        image_raw = Image.open(study_view_filepath)
                        image_tensor = self.transform(image_raw).to(self.device)

                        # Dicom images in 16bits, while pngs are 8 bits to save space
                        image_tensor = 65535 * image_tensor
                        image_tensor = image_tensor.unsqueeze(0)

                        # Apply Forward pass in stages
                        image_tensor_norm = (image_tensor - 32767.5) / 32767.5
                        feature_map = self.image_encoder.features((image_tensor_norm))
                        features = self.image_encoder.avgpool(feature_map)

                        study_views_feature_vector.append(features.squeeze()) # each features vector has shape of [768]
                        
                    # perform the feature vector concatenation method
                    if self.config.dataset.config.concatenate_features_method == "maxpool":
                        # stack them on the first axis
                        stacked_embeddings = torch.stack(study_views_feature_vector, dim= 0)    # [n_files, 768]
                        
                        # Apply max pooling along the batch dimension
                        joint_embeddings, _ = torch.max(stacked_embeddings, dim=0)              # [768]

                    elif self.config.dataset.config.concatenate_features_method == "concat":
                        # NOTE: DON'T USE, not fully implemented
                        # NOTE: here the config.networks.image_features_dimension has to be changed, thus we can't
                        # use this approach unless we concat [0, 0, 0, .., 0] and make all embedding vectors has 
                        # same shape
                        joint_embeddings = torch.cat(study_views_feature_vector, dim=0)         # [n_files * 768]

                    elif self.config.dataset.config.concatenate_features_method == "stack":
                        joint_embeddings = torch.stack(study_views_feature_vector, dim=0)       # [n_files, 768]

                    elif self.config.dataset.config.concatenate_features_method == "avgpool":
                        stacked_embeddings = torch.stack(study_views_feature_vector, dim= 0)    # [n_files, 768]
                        joint_embeddings = torch.mean(stacked_embeddings, dim=0)                # [768]
                    
                    else:
                        raise ValueError("Not implemented feature vector concatenation method")
                    
                    # add the concat method to the export path
                    sample_export_path = os.path.join(self.export_dir, row['study_path'].split('2D_100micron/')[-1])
                    sample_export_pth_filename = os.path.join(sample_export_path, '{}.pth'.format(self._get_patient_id(path=row['study_path'])))
                    
                    # create the directory
                    create_directory_if_not_exists(sample_export_path)
                    
                    # detach the features from cuda device to allow pin_memory=True when other objects are not on cuda and avoid torch multiprocessing 'spawn'
                    joint_embeddings = joint_embeddings.detach().cpu()

                    # export the features as a .pth file
                    torch.save(joint_embeddings, sample_export_pth_filename)         

                except Exception as e:
                    failed_txt_filepath = os.path.join(self.export_dir, 'failed.txt')
                    with open(failed_txt_filepath, "a") as myfile:
                        myfile.write(row['study_path'] + '\n' + str(e) + '\n\n')

image_feature_extractor = ImageFeatureExtractor
study_feature_extractor = StudyFeatureExtractor