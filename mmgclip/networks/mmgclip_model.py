import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from prettytable import PrettyTable

from .network_controller import getNetworkClass
from .projection_controller import get_projection_head

from ..utils.logger import logger

class MMGCLIP(nn.Module):
    def __init__(self, 
                 config = None):
        super().__init__()

        assert config is not None, 'Error in initializing the model. Missing training config object.'
        self.config = config
        
        # checking if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("WARNING: No CUDA device is found. This may take significantly longer!")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define both image and text encoders
        if self.config.networks.image_encoder.name == "ResNet50Encoder":
            self.image_encoder  = getNetworkClass(self.config.networks.image_encoder.name)
            self.image_encoder = self.image_encoder(pretrained=True, image_features_dimension = self.config.networks.image_encoder.image_features_dimension).to(self.device)
            logger.info(f"Using {self.image_encoder.__class__.__name__}")

        # for the text/label/image description, they are tokenized in the dataloader
        self.text_encoder   = getNetworkClass(self.config.networks.text_encoder.name)(pretrained = self.config.tokenizer.config.tokenizer_name).to(self.device)
        
        # projection layer to learn the features and map them to same dimensions
        if self.config.projection.config.projection_name != "ZeroProjection":
            self.image_projection_layer = get_projection_head(self.config.projection.config.projection_name)(
                embedding_dim=self.config.networks.image_encoder.image_features_dimension, 
                projection_dim=self.config.projection.config.output_projection_dimension, 
                dropout = self.config.networks.dropout.config.dropout).to(self.device)
            
            self.text_projection_layer  = get_projection_head(self.config.projection.config.projection_name)(
                embedding_dim=self.text_encoder.model_output_dimension, 
                projection_dim=self.config.projection.config.output_projection_dimension, 
                dropout = self.config.networks.dropout.config.dropout).to(self.device)        
            logger.info(f"Embeddings are projected to {self.config.projection.config.output_projection_dimension} features using {self.config.projection.config.projection_name}.")
        else:
            self.image_projection_layer = None
            self.text_projection_layer  = None

        # temperature parameter which controls the range of the logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.networks.logit_temperature)).to(self.device)

    def count_parameters(self, model):
        '''Counts the model trainable parameters.
        
        Args:
            model: a model object.
        
        Returns:
            count (int): the total number of traiable parameters.
        '''
        # return sum(p.numel() for p in model.parameters() if p.requires_grad)
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        logger.info(f"\n{table}")
        logger.info(f"Total Trainable Params: {total_params}")
        return total_params

    def encode_images(self, batch):
        '''Encodes/returnes the extracted image features flattened from a shape of (n, 1, image_features_dimension, 1, 1) 
        to (n, image_features_dimension)
        
        Args:
            batch: a batch from the dataloader.

        Returns:
            image features (torch.tensor): a tensor with shape of (n, image_features_dimension)
        '''
        flattened_embeddings = torch.flatten(batch['image_features'].to(self.device), 1)

        if self.config.networks.image_encoder.name == "ResNet50Encoder":
            # if the encoder type is resnet50, the batch will contain the image and we pass it to the encoder
            return self.image_encoder(flattened_embeddings)

        # return the flattened features if the case any other model is used (features were extracted by a separate batch file)
        return flattened_embeddings

    def encode_text(self, batch, text_pooling = 'eos'):
        '''Extracts the text features using a text encoder and performs text pooling on the extracted features.
        
        Args:
            batch: a batch from the dataloader.
            text_pooling (str): text pooling method. Default is 'eos'.

        Returns:
            text features (torch.tensor): a tensor with shape of (n, model_output_dimension)
        '''
        # extracting text features from the encoder output `last_hidden_state`` 
        text_features = self.text_encoder(batch['text_tokens'].to(self.device)) # [n, image_features_dimension]

        # performing end of sentence (eos) pooling
        if text_pooling == 'eos':
            eos_token_indices = batch['text_tokens']['attention_mask'].sum(dim=-1) -1
            text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        else:
            raise NotImplementedError(f"{text_pooling} method is not implemented...")
        
        return text_features

    def forward(self, batch, **kwargs):
        '''Forward pass of the model training.'''
        # extract the features
        image_features = self.encode_images(batch)                      # [n, image_features_dimension]
        text_features  = self.encode_text(batch, text_pooling='eos')    # [n, model_output_dimension]

        # linear projection to map from each encoder’s representation to the multi-modal embedding space.
        image_embeddings = self.image_projection_layer(image_features) if self.image_projection_layer is not None else image_features  # [n, output_projection_dimension]
        text_embeddings  = self.text_projection_layer(text_features)   if self.text_projection_layer  is not None else text_features  # [n, output_projection_dimension]

        # normalise the embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True) # [n, output_projection_dimension]
        text_embeddings  = text_embeddings  / text_embeddings.norm(dim=1, keepdim=True)  # [n, output_projection_dimension]

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp() # clamp the logit_scale?
        # print(logit_scale)

        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t() # [n, n]
        logits_per_text  = logit_scale * text_embeddings @ image_embeddings.t() # [n, n]
        # logits_per_text  = logits_per_image.t()                               # [n, n]

        # After calculating logits
        # posteriors_per_image = F.softmax(logits_per_image, dim=1)
        # posteriors_per_text = F.softmax(logits_per_text, dim=1)

        # text_probs = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)
        # print(text_probs)

        output = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "logit_scale": logit_scale,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text
        }

        if self.config.loss.config.loss_name == "MMGCLIPLoss" and not kwargs.get('validation', False) == True:
            # in the exam-report implementation, we use the image-label dataset that doesn't have the text impression section
            # thus this block is not valid for it, any future modification of the exam-report validation has to ensure necessary modifications here
            # This so far is accessable during training with MMGCLIPLoss loss.
            
            # for experimenting with T2T TCL Loss component
            batch['text_tokens'] = batch['image_impression_tokens'] # to re-use the code with minimum modification
            text_features2  = self.encode_text(batch, text_pooling='eos')    # [n, model_output_dimension]
            text_embeddings2  = self.text_projection_layer(text_features2)     # [n, output_projection_dimension]
            text_embeddings2  = text_embeddings2  / text_embeddings2.norm(dim=1, keepdim=True)  # [n, output_projection_dimension]
            output['text_embeddings2'] = text_embeddings2

        return output
    
class PromptClassifier(nn.Module):
    '''
    A zero-shot classifier wrapper based on MMGCLIP model.
    '''
    def __init__(self, model = None):
        '''
        Initialized with both the model and the prompts input that is a dictionary of classes with their tokenizer output.
        It wraps the model outputs to a form of classification results.

        Args:
            model (object): an object of MMGCLIP model.

        Returns:
            outputs (dict): a dictionary of similarity scores and class names.
        '''
        super().__init__()
        self.model = model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = self.model.config.tokenizer.config.tokenizer_name)

    def forward(self, image_features, class_list, visualize=True, image_id = None, ground_truth = None):
        classes_similarities = []

        # structure the input
        inputs = {
            "image_features": image_features,
            "text_tokens": self.tokenizer(class_list, padding="max_length", truncation=True, return_tensors="pt", max_length=self.model.config.tokenizer.config.sequence_length)
        }

        self.model.eval()
        with torch.no_grad():
            # obtain the logits per image (image-text similarity score)
            # imgs are rows, texts are cols
            classes_similarities = self.model(inputs)['logits_per_image'] 

            # apply softmax to obtain normalized probabilities
            classes_similarities = classes_similarities.softmax(dim=-1)

        # structure the output
        outputs = {
            "classes_similarities": classes_similarities,
            "similarities_argmax": torch.argmax(classes_similarities, dim=-1)[0].item(),
            "class_list": class_list
        }

        if visualize:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from ..utils.data_utils import create_path
            from PIL import Image
            sns.set_theme(style="white", palette="rocket")

            assert image_id is not None, "For visualizing results, image_id value is required."

            plt.figure(figsize=(14, 12))

            y = np.arange(classes_similarities.detach().cpu().shape[-1])
            view_path = create_path(image_id, base_dataset_path=self.model.config.dataset.config.base_dataset_path)
            view_img = Image.open(view_path)

            plt.subplot(2, 2, 1)
            plt.imshow(view_img, cmap='gray')
            plt.axis('off')
            plt.xlabel(image_id)

            for idx, value in enumerate(class_list):
                prob = classes_similarities.detach().cpu().numpy().flatten()[idx]

                plt.subplot(2, 2, 2)
                plt.grid(False)
                plt.barh(y[idx], prob)
                plt.gca().invert_yaxis()
                plt.gca().set_axisbelow(True)

                plt.yticks(y, class_list)
                plt.xlabel("probability")

            plt.title(f"TP: {ground_truth}") if ground_truth else None
            plt.subplots_adjust(wspace=0.5)
            plt.show()
        
        return outputs
            

