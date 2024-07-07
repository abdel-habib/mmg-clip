import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import time
import uuid
import numpy as np

from ..utils.logger import logger, plot_logits_tensorboard
from ..utils.train_utils import epoch_time
from ..utils.global_utils import seeding
from ..networks.mmgclip_model import MMGCLIP as model
from ..loss.loss_controller import create_loss
from ..scheduler.warmup_cosine import LinearWarmupCosineAnnealingLR
from ..callbacks.early_stopping import EarlyStopper
from ..utils.global_utils import create_directory_if_not_exists
from ..evaluator import Evaluator
from ..prompts.enums import MassShapeLabels, BenignMalignantDatasetLabels

from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

class ClassifierExperiment:
    """
    This class implements the basic life cycle for a classification task.
    The basic life cycle of a ClassifierExperiment is:

        run():
            for epoch in max_epochs:
                train()
                validate()
        test()
    """
    def __init__(self,
                 config = None,
                 train_dataloader = None,
                 valid_dataloader = None,
                 test_dataloader = None,
                 tokenizer = None):
        
        super().__init__()

        # time tracking 
        self._time_start = None
        self._time_end = None

        # data loaders
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        # tokenizer
        self.tokenizer = tokenizer

        self.config = config
        self.current_epoch = 0

        logger.info(f"Experiment Parameters: name={self.__class__.__name__}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # validate the config passed
        assert self.config is not None, 'Error in initializing the model. Missing training config object.'

        # init the model
        self.model = model(config=config).to(self.device)
        self.model.count_parameters(self.model)
        # logger.info(f"Total Trainable Parameters: {self.model.count_parameters(self.model)}")

        # init the loss
        self.criterion = create_loss(self.config.loss.config.loss_name)().to(self.device)
        logger.info(f"Using {self.criterion.__class__.__name__} loss.")

        # init the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.optimizer.config.learning_rate, weight_decay=self.config.optimizer.config.weight_decay)

        # init the lr scheduler
        if self.config.scheduler.name == "cosine":
            self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, total_steps=self.config.scheduler.config.epochs, warmup_steps=self.config.scheduler.config.warmup_epochs)
        elif self.config.scheduler.name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.config.scheduler.config.patience, verbose=True)

        logger.info(f"Using {self.scheduler.__class__.__name__}")

        # init the early stopper
        self.ckp_path = create_directory_if_not_exists(self.config.checkpoints.checkpoints_export_dir)
        self.ckp_path = os.path.join(self.ckp_path, self.config.checkpoints.checkpoints_file_name)
        self.early_stopper = EarlyStopper(patience=self.config.base.patience, delta=0, trace_func=logger.warning)

        # tensorboard config
        self.writer = SummaryWriter(log_dir=self.config.base.tensorboard_export_dir)


    def train(self):
        '''This method is excuted once per epoch.'''
        
        # Set the model to training mode
        self.model.train()

        # loss epoch placeholder
        loss_list = []

        # Iterate over the training data
        # the batch is a dict of multiple keys and values, can be seen from `collate_fn` used by the dataloader
        for index, batch in enumerate(self.train_dataloader):
            # Zero the gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass
            outputs = self.model(batch)
            
            # compute the loss
            loss, labels = self.criterion(**outputs)

            # backward pass
            loss.backward()

            # update the weights
            self.optimizer.step()

            # accumulate the epoch losses
            loss_list.append(loss.item())

            # logger.info(f"Epoch: {self.current_epoch+1} Train batch {index + 1} loss: {loss.item()}, {100*(index+1)/len(self.train_dataloader):.1f}% complete")
        
        # update the lr scheduler
        self.scheduler.step()

        epoch_loss = np.mean(loss_list)

        self.writer.add_scalar('loss/train', epoch_loss, self.current_epoch+1)

        return epoch_loss

    def validate(self):
        ''''
        This method runs validation cycle. Note that model 
        needs to be switched to eval mode and no_grad needs to 
        be called so that gradients do not propagate
        '''
        # Set the model to eval mode
        self.model.eval()

        # placeholders
        loss_list = []
        
        if 'BenignMalignantDatasetLabels' in self.config.experiments.config.metrics:
            malignancy_tokens   = self.tokenizer(['Finding suggesting malignant.'], padding="max_length", truncation=True, return_tensors="pt", max_length=self.model.config.tokenizer.config.sequence_length)
            predictions_malig   = []
            targets_malig       = []

        if 'MassShapeLabels' in self.config.experiments.config.metrics:
            shapes_list         = [f"Mass shape is {label.name}." for label in MassShapeLabels]
            shapes_tokens       = self.tokenizer(shapes_list, padding="max_length", truncation=True, return_tensors="pt", max_length=self.model.config.tokenizer.config.sequence_length)
            targets_shapes      = []
            predictions_shapes  = []

        if 'birads' in self.config.experiments.config.metrics:
            birads_list         = [f"BIRADS score of {i}." for i in range(0, 7)]
            birads_list.insert(0, "BIRADS unknown.")
            birads_tokens       = self.tokenizer(birads_list, padding="max_length", truncation=True, return_tensors="pt", max_length=self.model.config.tokenizer.config.sequence_length)
            targets_birads      = []
            predictions_birads  = []

        # Disable gradient calculation
        with torch.no_grad():
            for index, batch in enumerate(self.valid_dataloader):

                # Forward pass
                outputs = self.model(batch)

                # compute the loss
                loss, labels = self.criterion(**outputs)

                # accumulate the epoch losses
                loss_list.append(loss.item())

                # to compute the aucroc for malignancy as accuracy metric
                prompt_labels = batch['prompt_labels']

                if 'BenignMalignantDatasetLabels' in self.config.experiments.config.metrics:
                    # MALIG AUC
                    if isinstance(prompt_labels[0]['BenignMalignantDatasetLabels'], int):
                        # exam-report dataset, this is passed as an int
                        y_true_malig = [label['BenignMalignantDatasetLabels'] for label in prompt_labels]
                    elif isinstance(prompt_labels[0]['BenignMalignantDatasetLabels'], str):
                        # image-label dataset, this is passed as label str 
                        y_true_malig = [BenignMalignantDatasetLabels[label['BenignMalignantDatasetLabels']].value for label in prompt_labels]

                    targets_malig.extend(y_true_malig) # y_true_malig

                    # we change the batch input to 'malignant'
                    val_outputs = self.model({
                        "image_features": batch['image_features'],
                        "text_tokens": malignancy_tokens
                    })

                    similarities_malig = val_outputs['logits_per_image'].detach().cpu().numpy()
                    predictions_malig.extend(similarities_malig) # y_pred

                if 'MassShapeLabels' in self.config.experiments.config.metrics:
                    # SHAPES AUC
                    if isinstance(prompt_labels[0]['MassShapeLabels'], int):
                        y_true_shapes = [label['MassShapeLabels'] for label in prompt_labels]
                    elif isinstance(prompt_labels[0]['MassShapeLabels'], str):
                        y_true_shapes = [MassShapeLabels[label['MassShapeLabels']].value for label in prompt_labels]

                    targets_shapes.extend(y_true_shapes) # y_true_shapes

                    # we change the batch input to 'malignant'
                    val_outputs = self.model({
                        "image_features": batch['image_features'],
                        "text_tokens": shapes_tokens
                    }) 
                    similarities_shapes = val_outputs['logits_per_image'].detach().cpu().numpy()
                    predictions_shapes.extend(similarities_shapes) # y_pred

                if 'birads' in self.config.experiments.config.metrics:
                    # BIRADS AUC
                    y_true_birads = [-1 if label['BIRADS'] == "unknown" else int(label['BIRADS']) for label in prompt_labels] # replace `unknown` with label -1, and cast to int
                    targets_birads.extend(y_true_birads) # y_true_malig

                    # we change the batch input to 'malignant'
                    val_outputs = self.model({
                        "image_features": batch['image_features'],
                        "text_tokens": birads_tokens
                    })
                    
                    similarities_birads = val_outputs['logits_per_image'].detach().cpu().numpy()
                    predictions_birads.extend(similarities_birads) # y_pred

        # obtain the metrics
        epoch_loss = np.mean(loss_list)
        self.writer.add_scalar('loss/val', epoch_loss, self.current_epoch+1)
        epoch_auc_malig, epoch_auc_shapes, epoch_auc_birads, auc_list = None, [], [], []

        if 'BenignMalignantDatasetLabels' in  self.config.experiments.config.metrics:
            predictions_malig = [pred[0] for pred in predictions_malig]
            
            fpr, tpr, thresholds = metrics.roc_curve(targets_malig, predictions_malig)
            epoch_auc_malig  = metrics.auc(fpr, tpr)
            self.writer.add_scalar('auc/val/malig', epoch_auc_malig, self.current_epoch+1)
            auc_list.append(epoch_auc_malig)

        if 'MassShapeLabels' in self.config.experiments.config.metrics:
            predictions_shapes = np.array(predictions_shapes)
            
            for idx, value in enumerate(shapes_list):
                # change roc_curve to roc_auc_score
                # roc = metrics.roc_auc_score(np.array(targets_shapes) == idx, predictions_shapes[:, idx])
                fpr, tpr, thresholds = metrics.roc_curve(np.array(targets_shapes) == idx, predictions_shapes[:, idx])
                roc = metrics.auc(fpr, tpr)
                epoch_auc_shapes.append(roc)

            # average auc for multi-class
            epoch_auc_shapes = np.mean(epoch_auc_shapes)
            self.writer.add_scalar('auc/val/shapes', epoch_auc_shapes, self.current_epoch+1)
            auc_list.append(epoch_auc_shapes)

        if 'birads' in self.config.experiments.config.metrics:
            predictions_birads = np.array(predictions_birads)

            for idx, value in enumerate(birads_list):
                # to allow validating unknown (-1) label when comparing with the target
                # roc = metrics.roc_auc_score(np.array(targets_birads) == idx - 1, predictions_birads[:, idx])
                fpr, tpr, thresholds = metrics.roc_curve(np.array(targets_birads) == idx - 1, predictions_birads[:, idx])
                roc = metrics.auc(fpr, tpr)
                epoch_auc_birads.append(roc)

            # average auc for multi-class
            epoch_auc_birads = np.mean(epoch_auc_birads)
            self.writer.add_scalar('auc/val/birads', epoch_auc_birads, self.current_epoch+1)
            auc_list.append(epoch_auc_birads)
            
        if len(auc_list) > 1:            
            epoch_auc_means = np.mean(auc_list)
            self.writer.add_scalar('auc/val/average', epoch_auc_means, self.current_epoch+1)

        # last batch for every epoch
        # plot_logits_tensorboard(logits_per_image=outputs['logits_per_image'],
        #                                 logits_per_text=outputs['logits_per_text'],
        #                                 writer=self.writer,
        #                                 global_step= self.current_epoch + 1,
        #                                 suptitle=f"Epoch {self.current_epoch+1}, Batch {index + 1}")

        return epoch_loss, \
            epoch_auc_malig if 'BenignMalignantDatasetLabels' in  self.config.experiments.config.metrics else -1, \
            epoch_auc_shapes if 'MassShapeLabels' in self.config.experiments.config.metrics else -1, \
            epoch_auc_birads if 'birads' in self.config.experiments.config.metrics else -1, \
            epoch_auc_means if len(self.config.experiments.config.metrics) > 1 else -1

    def test(self):
        '''This runs a test cycle on the test dataset.
        Computes various metrics and generates a confusion 
        matrix and ROC curve.
        '''
        logger.info("Running testing evaluator script.")

        Evaluator(config=self.config, 
                  test_dataloader=self.test_dataloader, 
                  tokenizer=self.tokenizer,
                  model=self.model).evaluate_experiment()

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end.
        """
        self._time_start = time.time()
        logger.info("Classifier training experiment started.")

        for self.current_epoch in range(self.config.scheduler.config.epochs):
            start_time = time.time()

            train_loss = self.train()
            valid_loss, valid_malig_auc, valid_shapes_auc, valid_birads_auc, valid_mean_auc = self.validate()
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # get the new lr for logging
            after_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar('lr', after_lr, self.current_epoch+1)

            # handle the early stopping and model export
            self.early_stopper(
                validation_loss=valid_loss,
                epoch=self.current_epoch,
                model=self.model,
                optimizer=self.optimizer,
                path= self.ckp_path
            )

            logger.info(f'Epoch: {self.current_epoch+1}/{self.config.scheduler.config.epochs} | epoch time: {epoch_mins}m {epoch_secs:.04}s | lr: {after_lr:.6f} | train/loss: {train_loss:.4f} | val/loss: {valid_loss:.4f} | val/auc/malig: {valid_malig_auc:.4f} | val/auc/shapes: {valid_shapes_auc:.4f} | val/auc/birads: {valid_birads_auc:.4f} | val/auc/mean: {valid_mean_auc:.4f}.')

            if self.early_stopper.early_stop:
                logger.warning(f"Early stopping triggered at epoch {self.current_epoch+1}. Ending model training.")
                self.writer.close()
                break

        # run test
        if len(self.config.dataset.eval.enum_classes) > 0 and self.test_dataloader is not None:
            self.test()

        self._time_end = time.time()
        logger.info(f"Experiment complete. Total time (H:M:S): {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
        self.writer.close()


