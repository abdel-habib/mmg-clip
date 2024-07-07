import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Union
from prettytable import PrettyTable

from scipy.special import softmax
from sklearn import metrics
from .utils.global_utils import logger
from .networks.mmgclip_model import MMGCLIP
from .loss.losses import CLIPLoss
from .utils.global_utils import create_directory_if_not_exists
from .utils.data_utils import process_class_list
from .prompts.enums import * # import all enums, get the required using globals().get('...')

sns.set_theme(style="white", palette="coolwarm")

class Evaluator:
    def __init__(self, 
                 config, 
                 test_dataloader = None,
                 tokenizer = None,
                 model = None,
                 cnn_eval = False):
        super().__init__()

        logger.info("Running evaluator on test split.")
        
        # load the config and device
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # dataset assertion
        assert test_dataloader is not None, "Evaluation requires a test dataloader object."
        self.test_dataloader = test_dataloader
        
        if not cnn_eval:
            # checkpoint assertion
            self.ckp_path = os.path.join(self.config.checkpoints.checkpoints_export_dir, self.config.checkpoints.checkpoints_file_name)
            print(f"ckp_path: {self.ckp_path}")
            assert os.path.isfile(self.ckp_path), f'Checkpoint file path "{self.ckp_path}" does not exist.'
            self.ckp_file = torch.load(self.ckp_path)
            logger.info(f"Loading model from {self.ckp_path}...")

            # load the model
            if model:
                logger.info("Using trained model instance...")
                self.model = model
            else:
                logger.info("Loading model from checkpoint...")
                self.model = MMGCLIP(config=self.config).to(self.device)
                self.model.load_state_dict(self.ckp_file['model_state_dict'])
                logger.info("Model loaded...")

        else:
            logger.info("Evaluating CNN, use evaluate_cnn method.")
        
        self.tokenizer = tokenizer

        # prepare output dir
        create_directory_if_not_exists(self.config.base.results_export_dir)
        
    def encode_text(self, text_tokens: Union[str, List[str], Dict, torch.Tensor]):
        if isinstance(text_tokens, str) or isinstance(text_tokens, list):
            tokens = self.tokenizer(
                text_tokens, padding="longest", truncation=True, return_tensors="pt", max_length=self.config.tokenizer.config.sequence_length
            )
            text_tokens = dict()

            text_tokens['text_tokens'] = tokens

        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens, text_pooling = 'eos')
            text_embeddings = self.model.text_projection_layer(text_embeddings) if self.config.projection.config.projection_name != "ZeroProjection" else text_embeddings
            text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=1, keepdim=True)
        return text_embeddings.detach().cpu().numpy()
    
    def encode_image(self, batch: dict):
        with torch.no_grad():
            image_embeddings = self.model.encode_images(batch)
            image_embeddings = self.model.image_projection_layer(image_embeddings) if self.config.projection.config.projection_name != "ZeroProjection" else image_embeddings
            image_embeddings = image_embeddings / torch.norm(image_embeddings, dim=1, keepdim=True)
        return image_embeddings.detach().cpu().numpy()
    
    def calculate_ci(self, scores):
        sorted_scores = np.sort(scores)
        lower_bound = sorted_scores[int(0.025 * len(sorted_scores))]
        upper_bound = sorted_scores[int(0.975 * len(sorted_scores))]
        return np.mean(scores), lower_bound, upper_bound

            
    # def zeroshot_eval_single(self, image_embeddings, label_names, classes_dict, key):
    #     logger.info("Evaluating zero-shot configuration.")
    #     # obtain the unique label names
    #     label_names = [[label] for label in label_names] # y_true
        
    #     # create a table
    #     results = PrettyTable(["Class", "AUROC", "Accuracy", "F1"])

    #     plt.figure()
    #     for idx, class_name in enumerate(classes_dict.keys()):

    #         # we evaluate each class independently, ex: class name = benign, thus prompts [no benign, benign]
    #         # the true label will always be index 1, that is pointing to benign label that the model was trained on
    #         prompts = [f'No {class_name}', f'{class_name}']  

    #         # pass to the tokenizer and encoder
    #         text_embeddings = self.encode_text(prompts)

    #         # compute the cosine similarity between the ground truth text embeddings and the image embeddings output from the model encoder
    #         similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)

    #         # obtain a normalized probabilities/predictions
    #         similarities = softmax(similarities, axis=1)

    #         # generate the y_true, label 1 will always be given to the class name of index 1 that the model was trained on
    #         y_true = [1 if class_name in label else 0 for label in label_names]

    #         # obtain the metrics
    #         fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, 1])
    #         roc  = metrics.auc(fpr, tpr)
    #         accuracy = metrics.accuracy_score(y_true, np.argmax(similarities, axis=1))
    #         f1score = metrics.f1_score(y_true, np.argmax(similarities, axis=1))
            
    #         results.add_row([class_name, roc, accuracy, f1score])

    #         # plot the roc per class
    #         plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.2f)' % (class_name, roc))
        
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    #     plt.show()

    #     # Save the plot
    #     plt.savefig(self.ckp_path.replace('checkpoints', 'results').replace('.pth', f'_{key}_classwise_roc.png'))
    #     return results
    
    def clf_conf_matrix(self, image_features, label_names, classes_dict, key, use_logits = True):
        '''
        This function generates a confusion matrix from the network, where it receives all text prompts at once in their original format
          at behaves like a traditional CNN classifier.

        Code to generate such ROC curves has been commented as it was not used in the study or previous work. 
        '''
        logger.info(f"Evaluating prompt classifier for {key}.")
    
        #  obtainthe unique label names
        label_names = [[label[key]] for label in label_names] # y_true

        # obtain the y_true from the dataset
        y_true = np.array([classes_dict[label[0].replace(' ', '').replace('-', '')] for label in label_names])

        # create a table
        results = PrettyTable(["Class", "AUROC", "Accuracy", "F1"])

        # obtain the classes labels as prompts
        label_names = [process_class_list(label_name_list) for label_name_list in label_names]
        classes_prompts = process_class_list(list(classes_dict.keys()))

        # remove unknown class if it exist, assuming that all the data samples has known BI-RAD label
        # might need to ensure that all test samples has known labels
        classes_prompts.remove('unknown') if 'unknown' in classes_prompts else None
        
        # structure the input
        inputs = {
            "image_features": image_features,
            "text_tokens": self.tokenizer(classes_prompts, padding="max_length", truncation=True, return_tensors="pt", max_length=self.model.config.tokenizer.config.sequence_length)
        }

        # obtain the logits per image (image-text similarity score)
        # imgs are rows, texts are cols
        with torch.no_grad():
            similarities = self.model(inputs, validation=True)['logits_per_image']

            # apply softmax to obtain normalized probabilities
            similarities = similarities.softmax(dim=-1)

        # apply argmax to find the highest probability predictions
        y_pred = torch.argmax(similarities, dim=-1).detach().cpu().numpy()

        # cast similarities to numpy
        similarities = similarities.detach().cpu().numpy()

        # to plot only the confusion matrix, rest of the code is for generating a class-wise roc and confusion matrix aside
        plt.figure(figsize=(8, 6))
        conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=range(len(classes_prompts)))
        ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes_prompts, yticklabels=classes_prompts)
        plt.title('Confusion Matrix')
        # Adjust the tick labels alignment
        ax.set_xticklabels(ax.get_xticklabels(), ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), va='center')

        plt.show()

        # # Create a 1x2 subplot for confusion matrix and ROC curve
        # fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # # Generate confusion matrix and plot it
        # conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=range(len(classes_prompts)))
        # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(classes_prompts)), yticklabels=range(len(classes_prompts)), ax=axes[0])
        # axes[0].set_title('Confusion Matrix')
        # axes[0].set_xticklabels(classes_prompts)
        # axes[0].set_yticklabels(classes_prompts)

        # # construct the results
        # results = {}

        # if len(classes_prompts) > 2:
        #     # multi-class one-vs-all ROC
        #     for idx, value in enumerate(classes_prompts):
        #         results[value] = {}

        #         # change roc_curve to roc_auc_score
        #         roc = metrics.roc_auc_score(y_true == idx, similarities[:, idx])
        #         fpr, tpr, thresholds = metrics.roc_curve(y_true == idx, similarities[:, idx])
        #         axes[1].plot(fpr, tpr, label=f'Class {value} (AUC = {roc:.4f}')

        #         results[value]['auc'] = roc

        #         # accuracy per class
        #         class_pred = y_pred == idx
        #         results[value]['accuracy'] = np.mean(class_pred == (y_true == idx))

        # else:
        #     # Here Malignant is treated as label 1 as in the enum, and the binary ROC is based on its similarity values.
        #     roc = metrics.roc_auc_score(y_true, y_pred)
        #     fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, 1])
        #     axes[1].plot(fpr, tpr, label=f'{classes_prompts[0]} vs {classes_prompts[1]} AUC = {roc:.4f}')
        #     results['auc'] = roc

        # axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        # axes[1].set_xlabel('False Positive Rate')
        # axes[1].set_ylabel('True Positive Rate')
        # axes[1].set_title('ROC Curve')
        # axes[1].legend()

        # accuracy = metrics.accuracy_score(y_true, y_pred)
        # f1score = metrics.f1_score(y_true, y_pred, average='binary' if len(classes_prompts) <= 2 else 'micro')

        # results['accuracy'] = accuracy
        # results['f1score'] = f1score

        # Save the plot
        plt.savefig(os.path.join(self.config.base.results_export_dir, 'classifier', f'model_{key}_confusion_matrix.png'))
        # plt.savefig(self.ckp_path.replace('checkpoints', 'results/classifier').replace('.pth', f'_{key}_classwise_roc.png'))

        return results

    def zeroshot_eval(self, image_embeddings, label_names, classes_dict, key, use_logits = True):
        logger.info(f"Evaluating zero-shot prompt configuration for {key}.")

        #  obtainthe unique label names
        label_names = [process_class_list([label[key]]) for label in label_names] # y_true
        
        classes_prompts = process_class_list(list(classes_dict.keys()))

        # create a table
        results = PrettyTable(["Class", "AUROC", "Accuracy", "F1"])

        plt.figure()
        for idx, class_name in enumerate(classes_prompts):

            # we evaluate each class independently, ex: class name = benign, thus prompts [no benign, benign]
            # the true label will always be index 1, that is pointing to benign label that the model was trained on
            prompts = [f'No {class_name}', f'{class_name}']

            # pass to the tokenizer and encoder
            text_embeddings = self.encode_text(prompts)

            # compute the cosine similarity between the ground truth text embeddings and the image embeddings output from the model encoder
            if use_logits:
                # similarities here are the logits per image same as the computation of the model forward pass
                logit_scale = self.model.logit_scale.exp()
                logit_scale = logit_scale.detach().cpu().numpy()

                similarities = logit_scale * image_embeddings @ np.transpose(text_embeddings)
            else:
                similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings) # [samples, n_classes]

            # obtain a normalized probabilities/predictions
            similarities = softmax(similarities, axis=1)

            # generate the y_true, label 1 will always be given to the class name of index 1 that the model was trained on
            y_true = np.array([1 if class_name in label else 0 for label in label_names])

            # obtain the metrics
            fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, 1])
            roc  = metrics.auc(fpr, tpr)
            
            accuracy = metrics.accuracy_score(y_true, np.argmax(similarities, axis=1))
            f1score = metrics.f1_score(y_true, np.argmax(similarities, axis=1))
            
            results.add_row([class_name, roc, accuracy, f1score])

            # plot the roc per class
            plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.4f)' % (class_name, roc))
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # Save the plot
        plt.savefig(os.path.join(self.config.base.results_export_dir, 'zeroshot', f'model_{key}_classwise_roc.png'))
        # plt.savefig(self.ckp_path.replace('checkpoints', 'results/zeroshot').replace('.pth', f'_{key}_classwise_roc.png'))
        return results
    
    def zeroshot_label_prompt(self, image_embeddings, label_names, classes_dict, key, use_logits = True):
        logger.info(f"Evaluating zero-shot prompt configuration for {key}.")

        #  obtainthe unique label names
        label_names = [process_class_list([label[key]]) for label in label_names] # y_true
        
        classes_prompts = process_class_list(list(classes_dict.keys()))

        # create a table
        results = PrettyTable(["Class", "AUROC", "Accuracy", "F1"])
        
        # get the specific label sentence
        if key == "BenignMalignantDatasetLabels":
            prompts = [f"Finding suggesting {label}." for label in classes_prompts]
        elif key == "MassShapeLabels":
            prompts = [f"Mass shape is {label}." for label in classes_prompts]
        elif key == "MassMarginLabels":
            prompts = [f"Mass margin is {label}." for label in classes_prompts]
        elif key == "HasMassLabels":
            prompts = ["No mass was observed.", "Findings revealed a mass."]
        elif key == "HasArchDistortion":
            prompts = ["Normal architecture is visible.", "Displayed architectural distortion."]
        elif key == "HasCalcification":
            prompts = ["No calcifications are present.", "Finding suggesting calcifications."]

        plt.figure()

        # pass to the tokenizer and encoder
        text_embeddings = self.encode_text(prompts)

        # compute the cosine similarity between the ground truth text embeddings and the image embeddings output from the model encoder
        if use_logits:
            # similarities here are the logits per image same as the computation of the model forward pass
            logit_scale = self.model.logit_scale.exp()
            logit_scale = logit_scale.detach().cpu().numpy()

            similarities = logit_scale * image_embeddings @ np.transpose(text_embeddings)
        else:
            similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings) # [samples, n_classes]

        # obtain a normalized probabilities/predictions
        similarities = softmax(similarities, axis=1)

        # obtain y-true from the dataset
        y_true = np.array([classes_dict[label[0].replace(' ', '').replace('-', '')] for label in label_names])

        # apply argmax to find the highest probability predictions
        y_pred = np.argmax(similarities, axis=-1) #torch.argmax(similarities, dim=-1).detach().cpu().numpy()

        # construct the results
        results = {}

        # placeholder for mean roc
        roc_curves = []

        for idx, value in enumerate(prompts):
            results[value] = {}

            # change roc_curve to roc_auc_score
            roc = metrics.roc_auc_score(y_true == idx, similarities[:, idx])
            fpr, tpr, thresholds = metrics.roc_curve(y_true == idx, similarities[:, idx])
            plt.plot(fpr, tpr, label=f'{value} (AUC = {roc:.4f})')

            results[value]['auc'] = roc

            roc_curves.append((fpr, tpr))

            # accuracy per class
            class_pred = y_pred == idx
            results[value]['accuracy'] = np.mean(class_pred == (y_true == idx))
        
        # creating an average roc curve
        # Interpolation
        mean_fpr = np.linspace(0, 1, 100)

        tprs = []
        for fpr, tpr in roc_curves:
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        # Calculate mean and std
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        # Calculate AUC
        mean_auc = metrics.auc(mean_fpr, mean_tpr)

        # Plotting roc curves
        plt.plot(mean_fpr, mean_tpr, color='r', linewidth=2, label=f'Mean ROC (AUC = {mean_auc:.4f})')
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='lightcoral', alpha=0.3, label='± 1 std. dev.')

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        # Save the plot
        plt.savefig(os.path.join(self.config.base.results_export_dir, 'zeroshot_label_prompt', f'model_{key}_classwise_roc.png'))

        # Bootstrapping and CI for binary tasks
        if len(prompts) == 2:
            plt.figure()

            # take the similarities of the first class, like malignancy, mass, ...
            y_pred_class_sim = similarities[:, 1]

            # Number of bootstrap iterations
            n_iterations = 1000

            # scores
            auc_iterations_scores = []

            for _ in range(n_iterations):
                # Resample with replacement
                indices = np.random.choice(len(y_pred_class_sim), len(y_pred_class_sim), replace=True)
                bootstrap_predictions = y_pred_class_sim[indices]
                bootstrap_labels = y_true[indices]

                # Check if both classes are present in the bootstrap sample
                unique_classes = np.unique(bootstrap_labels)
                if len(unique_classes) == 2:

                    # calculate the auroc for the bootstrapped sample
                    roc = metrics.roc_auc_score(bootstrap_labels == 1, bootstrap_predictions)

                    # append the auc for the sample
                    auc_iterations_scores.append(roc)

            auc_ci = self.calculate_ci(auc_iterations_scores)

            # Plot accuracy histogram
            plt.hist(auc_iterations_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f'Histogram of {n_iterations} bootstrapped samples, showing 95% CI')
            plt.xlabel('Score')
            plt.ylabel('Frequency')

            # Add 95% confidence interval lines
            plt.axvline(x=auc_ci[0], color='green', linestyle='-', linewidth=1)
            plt.axvline(x=auc_ci[1], color='red', linestyle='--', linewidth=1)
            plt.axvline(x=auc_ci[2], color='red', linestyle='--', linewidth=1)
           
            plt.text(auc_ci[0], plt.ylim()[1]*0.9, f'Mean: {auc_ci[0]:.4f}', color='green')
            plt.text(auc_ci[1], plt.ylim()[1]*0.8, f'Lower CI:\n{auc_ci[1]:.4f}', color='red')
            plt.text(auc_ci[2], plt.ylim()[1]*0.7, f'Upper CI:\n{auc_ci[2]:.4f}', color='red')

            # Save the figure
            plt.savefig(os.path.join(self.config.base.results_export_dir, 'zeroshot_label_prompt', f'model_{key}_auc_CI.png'))

            results['auc_ci_mean'] = auc_ci[0]
            results['auc_ci_lower'] = auc_ci[1]
            results['auc_ci_higher'] = auc_ci[2]

        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1score = metrics.f1_score(y_true, y_pred, average='binary' if len(classes_prompts) <= 2 else 'micro')

        results['accuracy'] = accuracy
        results['f1score'] = f1score
        return results
    
    # def ova_clf_eval(self, image_features, label_names, classes_dict, key):
    #     logger.info(f"Evaluating OVA ROC for {key}.")

    #     #  obtainthe unique label names
    #     label_names = [process_class_list([label[key]]) for label in label_names] # y_true

    #     # create a table
    #     results = PrettyTable(["Class", "AUROC"])

    #     # obtain the classes labels as prompts
    #     classes_prompts = process_class_list(list(classes_dict.keys()))

    #     # Create a 1x2 subplot for confusion matrix and ROC curve
    #     fig = plt.figure()

    #     # placeholder for mean roc
    #     roc_curves = []

    #     # iterate over all prompts/labels. obtain a roc for each separately
    #     for class_name in classes_prompts:
    #         class_prompt = [class_name]   
            
    #         # structure the input
    #         inputs = {
    #             "image_features": image_features,
    #             "text_tokens": self.tokenizer(class_prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=self.model.config.tokenizer.config.sequence_length)
    #         }

    #         # obtain the logits per image (image-text similarity score)
    #         # imgs are rows, texts are cols
    #         with torch.no_grad():
    #             # we plot the roc using the logits directly, no need for a softmax as the softmax will return [1] for each row sample
    #             similarities = self.model(inputs, validation=True)['logits_per_image']

    #             # cast similarities to numpy
    #             similarities = similarities.detach().cpu().numpy()

    #             # obtain the y_true from the dataset
    #             y_true = np.array([1 if class_name in label else 0 for label in label_names])

    #             # obtain the metrics
    #             fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, 0])
    #             roc  = metrics.auc(fpr, tpr)

    #             roc_curves.append((fpr, tpr))

    #             # plot the roc per class
    #             plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.4f)' % (class_name, roc))

    #             results.add_row([class_name, roc])

    #     # creating an average roc curve
    #     # Interpolation
    #     mean_fpr = np.linspace(0, 1, 100)

    #     tprs = []
    #     for fpr, tpr in roc_curves:
    #         tprs.append(np.interp(mean_fpr, fpr, tpr))

    #     # Calculate mean and std
    #     mean_tpr = np.mean(tprs, axis=0)
    #     std_tpr = np.std(tprs, axis=0)

    #     # Calculate AUC
    #     mean_auc = metrics.auc(mean_fpr, mean_tpr)

    #     # Plotting roc curves
    #     plt.plot(mean_fpr, mean_tpr, color='r', linewidth=2, label=f'Mean ROC (AUC = {mean_auc:.4f})')
    #     plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='lightcoral', alpha=0.3, label='± 1 std. dev.')

    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    #     plt.show()

    #     # Save the plot
    #     plt.savefig(os.path.join(self.config.base.results_export_dir, 'ova', f'model_{key}_ova_roc.png'))

    #     return results

    def evaluate_experiment(self):
        self.model.eval()

        # placeholders
        image_features = []
        
        # image_logits = []
        # text_logits = []

        image_embeddings = []
        text_embeddings = []

        # for evaluating zero-shot classification, we obtain this from the dataloader 'image_description', would act like classes names
        label_names = []

        # for the sentence training and label evaluation
        prompt_labels = []

        # all results
        experiments_results = []
       
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.test_dataloader)):

                # Forward pass
                # outputs = self.model(batch)

                # append information
                # image_logits.append(outputs['logits_per_image'].detach().cpu().numpy())       # outputs['logits_per_image'] shape [n, n] 
                # text_logits.append(outputs['logits_per_text'].detach().cpu().numpy())         # outputs['logits_per_text'] shape [n, n]
                label_names.extend(batch['image_description'])                                  # both labels and text description 
                image_features.extend(batch['image_features'])
                prompt_labels.extend(batch['prompt_labels'])
                image_embeddings.append(self.encode_image(batch))
                text_embeddings.append(self.encode_text(batch))

            # image_logits = np.concatenate(image_logits, axis=0)     # before concat shape [iterations, n, n], after concat shape [iterations x n, n]
            # text_logits = np.concatenate(text_logits, axis=0)       # before concat shape [iterations, n, n], after concat shape [iterations x n, n]
            
            image_embeddings = np.concatenate(image_embeddings, axis=0)
            text_embeddings = np.concatenate(text_embeddings, axis=0)

            image_features = torch.stack(image_features, dim=0) # (torch.Size([iterations x n, 1, 768, 1, 1]),) 

            for enum_class_name in self.config.dataset.eval.enum_classes:

                # get the dataset classes names 
                EnumClass = globals().get(enum_class_name)
                classes_dict = {label.name: label.value for label in EnumClass}

                if "zeroshot" in self.config.dataset.eval.method:
                    create_directory_if_not_exists(os.path.join(self.config.base.results_export_dir, 'zeroshot'))

                    # results = self.zeroshot_eval_single(image_embeddings=image_embeddings, label_names=label_names, classes_dict=classes_dict, key=enum_class_name)
                    results = self.zeroshot_eval(image_embeddings=image_embeddings, label_names=prompt_labels, classes_dict=classes_dict, key=enum_class_name)
                    
                    logger.info("Results From zero-shot evaluation...")
                    logger.info(f'\n{results}\n')
                    experiments_results.append(results)

                if "zeroshot_label_prompt" in self.config.dataset.eval.method:
                    create_directory_if_not_exists(os.path.join(self.config.base.results_export_dir, 'zeroshot_label_prompt'))

                    # results = self.zeroshot_eval_single(image_embeddings=image_embeddings, label_names=label_names, classes_dict=classes_dict, key=enum_class_name)
                    results = self.zeroshot_label_prompt(image_embeddings=image_embeddings, label_names=prompt_labels, classes_dict=classes_dict, key=enum_class_name)
                    
                    logger.info("Results From zero-shot label prompt evaluation...")
                    logger.info(f'\n{results}\n')
                    experiments_results.append(results)

                if "confustion_matrix" in self.config.dataset.eval.method:
                    create_directory_if_not_exists(os.path.join(self.config.base.results_export_dir, 'classifier'))

                    results = self.clf_conf_matrix(image_features=image_features, label_names=prompt_labels, classes_dict=classes_dict, key=enum_class_name)
                    
                    # logger.info("Results From label classifier evaluation...")
                    # logger.info(f'\n{results}\n')
                    # experiments_results.append(results)

                # if "ova" in self.config.dataset.eval.method:
                #     create_directory_if_not_exists(os.path.join(self.config.base.results_export_dir, 'ova'))

                #     results = self.ova_clf_eval(image_features=image_features, label_names=prompt_labels, classes_dict=classes_dict, key=enum_class_name)

                #     logger.info("Results From ova classifier evaluation...")
                #     logger.info(f'\n{results}\n')
                #     experiments_results.append(results)

            with open(os.path.join(self.config.base.results_export_dir, 'results.txt'), 'w') as file:
                for idx, result in enumerate(experiments_results):
                    file.write(str(result) + '\n\n')


    def evaluate_cnn(self, cnn):
        self.model = cnn

        # same as for evaluating zero-shot classification, we obtain this from the dataloader 'image_description', 
        # would act like classes names
        label_names = []

        # posteriors
        posteriors_list = []

        # create a table
        results = PrettyTable(["Class", "AUROC"])
        
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.test_dataloader)):
                # obtain the labels
                label_names.extend(batch['image_description'])

                # they are features output of avgpool
                image_features = batch['image_features']

                # Squeeze along the second dimension
                image_features = torch.squeeze(image_features, dim=1)  

                # obtain the logits
                logits = self.model.classifier(image_features)

                # obtain the posteriors
                posteriors = softmax(logits / 2)

                # append into a list
                posteriors_list.append(posteriors)

            # before concat shape [iterations, n, 2], after concat shape [iterations x n, 2]
            similarities = np.concatenate(posteriors_list, axis=0)     

            # we evaluate the cnn using only benign vs malignant enum 
            enum_class_name = self.config.dataset.eval.enum_classes[0]
            EnumClass = globals().get(enum_class_name)

            # get the dataset classes names 
            classes_dict = {label.name: label.value for label in EnumClass}

            # label names
            labels_names = list(classes_dict.keys())
            
            for idx, class_name in enumerate(labels_names):
                # obtain the y_true
                y_true = np.array([1 if class_name in label else 0 for label in label_names])

                # obtain the metrics
                fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, idx])
                roc  = metrics.auc(fpr, tpr)

                # plot the roc per class
                plt.plot(fpr, tpr, lw=2, label='%s (AUC = %0.4f)' % (class_name, roc))

                results.add_row([class_name, roc])

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

            # Save the plot
            create_directory_if_not_exists(os.path.join(self.config.base.results_export_dir, 'ova'))
            plt.savefig(os.path.join(self.config.base.results_export_dir, 'ova', f'model_cnn_{enum_class_name}_ova_roc.png'))

            return results
