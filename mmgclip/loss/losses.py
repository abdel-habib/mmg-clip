import torch
import torch.nn as nn 
import torch.nn.functional as F
from sentence_transformers import util

class CLIPLoss(nn.Module):
    '''Implementation of CLIP loss based on https://proceedings.mlr.press/v139/radford21a.
    
        @InProceedings{pmlr-v139-radford21a,
            title = 	 {Learning Transferable Visual Models From Natural Language Supervision},
            author =       {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
            booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
            pages = 	 {8748--8763},
            year = 	 {2021},
            editor = 	 {Meila, Marina and Zhang, Tong},
            volume = 	 {139},
            series = 	 {Proceedings of Machine Learning Research},
            month = 	 {18--24 Jul},
            publisher =    {PMLR},
            pdf = 	 {http://proceedings.mlr.press/v139/radford21a/radford21a.pdf},
            url = 	 {https://proceedings.mlr.press/v139/radford21a.html},
            abstract = 	 {State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.}
        }
    '''
    def __init__(self):
        super().__init__()

    def forward(self, logits_per_image, logits_per_text, **kwargs):
        '''
        Model outputs (input to the loss) could include other values beside the logits. This forward method extracts both logits and computes the final loss value.

        Returns:
            loss (tensor): loss value.
            labels (tensor): labels, that are range 0:n-1, where n is the batch size.
        '''
        n, _ = logits_per_image.shape # [n=batch_size]

        # symmetric loss function
        labels = torch.arange(n).cuda()
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        loss = (loss_i+loss_t)/2
        return loss, labels
    
class MMGCLIPLoss(nn.Module):
    '''Modified implementation of CXR-CLIP loss based on https://proceedings.mlr.press/v139/radford21a.
    
        @inproceedings{you2023cxr,
        title={Cxr-clip: Toward large scale chest x-ray language-image pre-training},
        author={You, Kihyun and Gu, Jawook and Ham, Jiyeon and Park, Beomhee and Kim, Jiho and Hong, Eun K and Baek, Woonhyuk and Roh, Byungseok},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={101--111},
        year={2023},
        organization={Springer}
        }
    '''
    def __init__(self, t2t_weight=0.5):
        super().__init__()

        self.t2t_weight = t2t_weight

    def forward(self, image_embeddings, text_embeddings, text_embeddings2, logit_scale, **kwargs):
        '''
        Model outputs (input to the loss) could include other values beside the logits. This forward method extracts both logits and computes the final loss value.

        Returns:
            loss (tensor): loss value.
            labels (tensor): labels, that are range 0:n-1, where n is the batch size.
        '''

        # CLIP Loss
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t() # [n, n]
        logits_per_text  = logit_scale * text_embeddings @ image_embeddings.t() # [n, n]

        n, _ = logits_per_image.shape # [n=batch_size]

        labels = torch.arange(n).cuda()
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        loss_clip = (loss_i+loss_t) / 2.0

        # TCL
        logits_per_t2t1 = logit_scale * text_embeddings2 @ text_embeddings.t()
        logits_per_t1t2 = logit_scale * text_embeddings @ text_embeddings2.t()

        loss_t2t1 = F.cross_entropy(logits_per_t2t1, labels)
        loss_t1t2 = F.cross_entropy(logits_per_t1t2, labels)

        loss_t2t = (loss_t2t1 + loss_t1t2) / 2.0

        # total loss
        loss = loss_clip + (loss_t2t * self.t2t_weight)

        return loss, labels

class AveragedMedicalCLIPLoss(nn.Module):
    '''
    A loss function that takes into account the similarity between the labels/text for 
    each batch independently using the cosine similarity of the text embeddings. This way, 
    texts with a similarity that satisfies a threshold are given the same label, where unique 
    texts are given unique labels.
    '''
    def __init__(self, similarity_threshold = 0.65):
        super().__init__()

        self.similarity_threshold = similarity_threshold

    def _mesaure_embeddings_similarity(self, embeddings):
        '''Measures the cosine similarity between the text embeddings (each and every one to all).
        
        Args:
            embeddings (tensor): represents the text embedding to measure its similarity with its self.
        
        Returns:
            cosine similarity scores (tensor): a matrix that have similairty between each embedding and the rest of values compared to.
        '''
        return util.cos_sim(embeddings, embeddings)
    
    def _assign_labels(self, cosine_sim_matrix, threshold=0.65):
        '''Assigns a unique label for each of the embeddings that have a similarity higher than or 
        equal to the threshold. If the condition is satisfied, similar labels will be assigned (in 
        the same location of the text embeddings computed on).

        For instance:
        >> cosine_scores = self._mesaure_embeddings_similarity(text_embeddings)
        >> print(consine_scores)
        >> tensor([[ 1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237],
                  [-0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000],
                  [ 1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237],
                  [-0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000],
                  [ 1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237],
                  [-0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000],
                  [ 1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237],
                  [-0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000, -0.0237,  1.0000]],
              device='cuda:0', grad_fn=<MmBackward0>)
        >> list_labels = self._assign_labels(cosine_scores, threshold=0.65)
        >> [0, 1, 0, 1, 0, 1, 0, 1]

        Args:
            cosine_sim_matrix (torch): the computed cosine similarity matrix using `_mesaure_embeddings_similarity`.
            threshold (int): the threshold value that assign embeddings similar labels.

        Returns:
            labels (list): list of labels for similar embeddings in their original index order.
        '''
        num_texts = cosine_sim_matrix.shape[0]
        labels = [-1] * num_texts  # Initialize labels with -1
        
        current_label = 0
        
        for i in range(num_texts):
            if labels[i] == -1:  # If the text hasn't been assigned a label yet
                labels[i] = current_label  # Assign it the current label
                for j in range(i+1, num_texts):
                    if cosine_sim_matrix[i][j] >= threshold:
                        if labels[j] == -1:
                            labels[j] = current_label  # Assign the same label if similarity >= threshold
                current_label += 1
        
        return labels
    
    def _average_logits(self, logits, list_labels):
        ''' Averages the logits on the same axis of the their unique labels. All columns (representing the logts of 
        the images batch to the text embeddings) with same label value in the labels list will be averaged togather.

        Args:
            logits (tensor): the logits obtained from the model.
            list_labels (list): list of labels for embeddings with similar meanings.

        Returns:
            averaged_logits_tensor (tensor): new averaged logits matrix, shape of (n, len(unique_labels))
        '''
        unique_labels = set(list_labels)
        averaged_logits = []

        for label in unique_labels:
            label_indices = [i for i, l in enumerate(list_labels) if l == label]
            label_logits = logits[:, label_indices]
            averaged_logit = torch.mean(label_logits, dim=1)
            averaged_logits.append(averaged_logit)

        averaged_logits_tensor = torch.stack(averaged_logits, dim=1)
        
        return averaged_logits_tensor


    def forward(self, image_embeddings, text_embeddings, logit_scale, logits_per_image, logits_per_text):
        '''
        Model outputs (input to the loss) could include other values beside the logits. This forward method extracts 
        both logits and computes the final loss value.

        Returns:
            loss (tensor): loss value.
            labels (tensor): labels, that are range 0:n-1, where n is the batch size.
        '''
        # measure the text embeddings similarities
        text_embeddings_similarity_matrix = self._mesaure_embeddings_similarity(text_embeddings)

        # assign a unique label to embeddings with similar similarity
        text_embeddings_similarity_labels = self._assign_labels(text_embeddings_similarity_matrix, threshold=self.similarity_threshold)

        # average the logits_per_image and obtain the logits_per_text from the transpose
        logits_per_image = self._average_logits(logits=logits_per_image, list_labels=text_embeddings_similarity_labels).to(logits_per_image.device)
        logits_per_text = logits_per_text.to(logits_per_text.device)

        # create a tensor from similarity labels
        labels = torch.tensor(text_embeddings_similarity_labels).to(logits_per_image.device)

        # symmetric loss function
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        loss = (loss_i+loss_t)/2
        return loss, labels








# class AveragedBinaryCLIPLoss(nn.Module):
#     '''
#     Modified implementation of CLIP loss. This averages the loss per class for each image, 
#     given that the labels could repeat for a binary classifier with batch size (n) > 2. 
    
#     Limitations:
#     1. Can't handle a 3 class problem. The averaging is for every even/odd columns.
#     2. If the unique classes > 2, it will fail as it will join those classes.
#     3. If the unique classes > batch size (n), it will fail as it will join those classes.
#     4. This loss expects the data to be equally balanced, if batch size = 2, then class 0, followed by class 1

#     Intermediate steps results n=8:

#     >> logits_per_image (input)

#                   text 1,  text 2,  text 1,  text 2, ..... 
#         tensor(
#         img1    [[-0.3695, -0.8987, -0.3323, -0.3540, -0.3375, -0.5998, -0.3583, -0.0797],
#         img2    [-0.9398, -1.1682, -0.9602, -0.7505, -1.0275, -0.5558, -0.3456, -0.3068],
#         img3    [-0.8346, -1.1233, -0.7055, -0.4546, -0.6598, -0.6412, -0.6927, -0.1958],
#         img4    [-0.8875, -1.3657, -0.6414, -0.8099, -0.8178, -0.8100, -0.6184, -0.1464],
#         img5    [-0.7839, -1.2652, -0.6129, -0.4527, -0.5410, -0.4618, -0.4844, -0.3835],
#         img6    [-1.0263, -1.3110, -0.7902, -0.7323, -0.6832, -0.9224, -0.6688, -0.6417],
#         img7    [-0.5663, -0.5041, -0.5145, -0.0413, -0.2905, -0.2322, -0.3936,  0.0914],
#         img8    [-0.1942, -0.7119, -0.3226, -0.1033, -0.2929, -0.1779, -0.2586, -0.1330]],
#         device='cuda:0', grad_fn=<MmBackward0>)

#     >> logits_per_image (.view(..))
#                   text 1,   text 2
#         tensor([[
#                 [-0.3695, -0.8987],
#                 [-0.3323, -0.3540],
#                 [-0.3375, -0.5998],
#                 [-0.3583, -0.0797]],

#                 [[-0.9398, -1.1682],
#                 [-0.9602, -0.7505],
#                 [-1.0275, -0.5558],
#                 [-0.3456, -0.3068]],
                
#                 .... ], device='cuda:0', grad_fn=<ViewBackward0>)

#     >> torch.mean(logits_per_image, dim=1)
#         tensor([[-0.3494, -0.4831],
#             [-0.8183, -0.6953],
#             [-0.7231, -0.6037],
#             [-0.7413, -0.7830],
#             [-0.6055, -0.6408],
#             [-0.7921, -0.9019],
#             [-0.4412, -0.1715],
#             [-0.2671, -0.2815]], device='cuda:0', grad_fn=<MeanBackward1>)

#     >> F.softmax(logits_per_image, dim=1)
#         tensor([[0.5334, 0.4666],
#             [0.4693, 0.5307],
#             [0.4702, 0.5298],
#             [0.5104, 0.4896],
#             [0.5088, 0.4912],
#             [0.5274, 0.4726],
#             [0.4330, 0.5670],
#             [0.5036, 0.4964]], device='cuda:0', grad_fn=<SoftmaxBackward0>)

#     >> probs_img[:, 1]
#         tensor([0.4666, 0.5307, 0.5298, 0.4896, 0.4912, 0.4726, 0.5670, 0.4964],
#             device='cuda:0', grad_fn=<SelectBackward0>)

#     >> loss
#         tensor(0.7441, device='cuda:0')

    
#     Returns:
#         loss (tensor): Averaged loss term per batch.
#     '''
#     def __init__(self):
#         super().__init__()

#     def forward(self, logits_per_image, logits_per_text, **kwargs):
#         '''
#         Model outputs (input to the loss) could include other values beside the logits. This forward method extracts both logits and computes the final loss value.

#         Returns:
#             loss (tensor): loss value.
#             labels (tensor): labels, that are range 0:n-1, where n is the batch size.
#         '''
#         n, _ = logits_per_image.shape # [n=batch_size]
        
#         # Works with only two classes, labelled 0 and 1 for binary task
#         n_classes = 2

#         # create a list of labels same size as the batch size (n), with labels 0 and 1 (n_classes - 1), shape [n]
#         labels = torch.arange(n) % (n_classes)
#         labels = labels.to(logits_per_image.device)

#         # reshape [n,n] -> [n, n//2, n_classes]
#         logits_per_image = logits_per_image.view(logits_per_image.shape[0], -1, 2)
#         logits_per_text = logits_per_text.view(logits_per_text.shape[0], -1, 2)
        
#         # Calculate mean along axis 1 to average columns [n, n_classes]
#         logits_per_image = torch.mean(logits_per_image, dim=1)
#         logits_per_text = torch.mean(logits_per_text, dim=1)

#         # Compute cross-entropy
#         loss_i = F.cross_entropy(logits_per_image, labels)
#         loss_t = F.cross_entropy(logits_per_text, labels)

#         loss = (loss_i+loss_t)/2

#         return loss, None

