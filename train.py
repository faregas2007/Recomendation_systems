from argparse import Namespace
from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import optuna
import mlflow
#from mlflow import pytorch
#from pprint import pformat

class Trainer(object):
    """Object used to facilitate training"""

    def __init__(self, 
        model, 
        device, 
        loss_fn=None, 
        optimizer=None, 
        scheduler= None, 
        trial=None,):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial

    def train_step(self, dataloader):
        """Train step

        Args:
            dataloader (torch.utils.data.DataLoader): torch dataloader to load batches from.
        """

        # Set model to train mode
        self.model.train()
        loss = 0.0

        size = len(dataloader.dataset)
        for batch, (user, item, label) in enumerate(dataloader):
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            # forward
            self.optimizer.zero_grad()
            prediction = self.model(user, item)

            # backward
            loss = self.loss_fn(prediction, label)
            loss.backward()
            self.optimizer.step()
            loss += loss.item() - loss

        return loss

    def eval_step(self, dataloader):
        """Validation or test step
        
        Args:
            dataloader(torch.utils.data.DataLoader): Torch dataloader to load batches from.
        """
        # Set model to eval mode
        self.model.eval()

        #size = len(dataloader.dataset)
        #num_batches = len(dataloader)
        loss = 0.0
        predictions, labels = [], []

        with torch.no_grad():
            for  batch, (user, item, label) in enumerate(dataloader):
                prediction = self.model(user, item)
                
                J = self.loss_fn(prediction, label).item()

                loss += (J - loss)/(batch + 1)
                #print(prediction.argmax(1), label)
                #correct += (prediction == label).type(torch.float).sum().item()
                # store outputs
                prediction = prediction.numpy()
                predictions.extend(prediction)
                labels.extend(label.numpy())

        return loss, np.vstack(labels), np.vstack(predictions)

    def predict_step(self, dataloader: torch.utils.data.DataLoader):
        """ Prediction (inference) step
        
        Note:
            Loss is not calcualted for this loop

        Args:
            dataloader (torch.utils.data.DataLoader): Torch dataloader to load batches from.
        """
        self.model.eval()
        predictions, labels = [], []

        # Interate over val batches
        with torch.no_grad():
            for batch, (user, item, label) in enumerate(dataloader):
                # Forward pass w/ inputs
                prediction = self.model(user, item)

                predictions.extend(prediction.numpy())
                labels.extend(label.numpy())
                
        return np.vstack(labels), np.vstack(predictions)

    def train(self, 
        num_epochs: int, 
        patience: int, 
        train_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: torch.utils.data.DataLoader):
        best_val_loss = np.inf
        best_model = None
        _patience = patience
        for epoch in range(num_epochs):
            # Step
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience # reset _patience
            else:
                _patience -= 1
            if not _patience: # 0
                print('Stopping early')
                break
            
            # Pruning based on the intermediate value
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            # Tracking
            mlflow.log_metrics(
                {'train_loss':train_loss, 'val_loss':val_loss}, step=epoch
            )

            # Logging
            print(
                f"Epoch:  {epoch + 1} |"
                f"train_loss: {train_loss:.5f},"
                f"val_loss: {val_loss:.5f},"
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E},"
                f"patience: {_patience}"
            )
        return best_model


def train(
    params_fp: Path,
    device: torch.device=torch.device('cpu'),
    trial: optuna.trial._trial.Trial=None,
)->Tuple:
    """
    Train a model

    Args:
        params (Namespace): parameters for data processing and training.
        train_dataloader (torch.utils.data.DataLoader): train data loader.
        val_dataloader (torch.utils.data.DataLoader): val data loader.
        model (nn.Module): Intitlaize model to train
        device (torch.device): Device to run model on
        trial (optuna.trial._trial.Trail, otpinal): Optuna optimization trial. Defaults to None.
    
    Returns:
        The best trained model loss and performance metrics
    """

    parans = Namespace(**utils.load_dict(params_fp))
    
    dataset = utls.get_data()
    dataloader = data.RCDataloader(params, dataset)
    train_dataloader = dataloader.get_train_set()
    test_dataloader = dataloader.get_test_set()
    
    model = models.initialize_model(params_fp, device)
    # Define loss
    loss_func = torch.nn.MSELoss()

    # Define optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.05, patience=5
    )

    # Trainer module
    trainer = Trainer(
        model = model,
        device = device,
        loss_fn = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        #trial = trial,
    )

    # Train
    best_val_loss, best_model = trainer.train(
        params.n_epochs, params.patience, train_dataloader, test_dataloader
    )

    return params, best_model, best_val_loss
