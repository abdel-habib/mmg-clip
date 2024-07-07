import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import torch
import os
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer

from ..utils.logger import logger
from .projection import LinearProjectionLayer

PROJECTION_HEAD_DIM = 512

class ConvNextTiny(nn.Module):
    '''
    ConvNext_Tiny pre-trained model for extracting image features. This class only loads pre-trained models 
    and returns the image features as an output.
    
    Args:
        from_pretrained (str): path to the pytorch model file.

    Returns:
        output (tensor): The output tensor from the model.
    '''

    def __init__(self):
        super().__init__()
        
        # define the model
        self.model = None

    def from_pretrained(self, model_path = None):
        assert os.path.isfile(model_path), "Model `.pt` file doesn't exist."

        self.model = torch.jit.load(model_path)

        return self.model

    def forward(self, x):
        '''
        Forward pass of the model.

        Args:
            x (tensor): Input tensor shape shape (B, C, H, W).

        Returns:
            output (tensor): The output tensor from the model.
        '''
        if self.model is None:
            assert ImportError("Model was not loaded correctly. Call `from_pretrained` and pass the model file path first.")

        x = self.model.features(x)
        x = self.model.avgpool(x) # ([1, 768, 1, 1])
        return x

class ResNet50Encoder(nn.Module):
    '''
    ResNet50 encoder with a projection head.
    
    Args:
        pretrained (bool): If True, the model will be initialized with a pre-trained model.
    
    Returns:
        output (tensor): The output tensor from the model.
    '''
    def __init__(self, pretrained=True, image_features_dimension = 768):
        super().__init__()
        logger.info(f"Initializing 'resnet50' as the image encoder.")

        # Load the pre-trained model
        self.model = models.resnet50(pretrained=pretrained)
        self.model_output_dimension = self.model.fc.in_features
        del self.model.fc

        # Disable gradients on all model parameters to freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected resnet layer with a new fc layer
        # self.model.fc = nn.Linear(self.model_output_dimension, image_features_dimension, bias=False)

        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

        # Unfreeze the last few layers of the model
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        '''
        Forward pass of the model.

        Args:
            x (tensor): Input tensor shape shape (B, C, H, W).

        Returns:
            output (tensor): The output tensor from the model.
        '''
        # If the data is 1D, we have to reshape it to 4D
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, 1, x.shape[1])
            x = x.repeat(1, 3, 1, 1)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.model.fc(x)

        return x

class BertEncoder(nn.Module):
    '''
    BERT Encoder.
    
    Args:
        pretrained (str): The name of the pre-trained model to use.
    
    Returns:
        output (tensor): The output tensor from the model.
    '''
    def __init__(self, pretrained = None):
        super().__init__()

        logger.info(f"Initializing pretrained `{pretrained}` as the text encoder and tokenizer.")
        
        # Load the pre-trained model and tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained)

        # Freeze BERT layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model_output_dimension = self.model.config.hidden_size

    def forward(self, x):
        '''
        Forward pass of the model.

        Args:
            x (tensor): Input tensor shape shape (B, C, H, W).

        Returns:
            output (tensor): The output tensor from the model.
        '''
        return self.model(**x)['last_hidden_state'] # has shape of (batch_size, sequence_length, hidden_size)