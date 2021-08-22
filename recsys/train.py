import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import optuna
import mlflow

import os
import numpy as np
from numpyencoder import NumpyEncoder

from argparse import Namespace
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import tempfile
import json

#from utils import *
#from config import *
#from data import *
#from models import *

from recsys import utils, config, data, models, eval

class Trainer(object):
    def __init__(self, 
        model, 
        device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        loss_fn=None, 
        optimizer=None, 
        scheduler= None, 
        trial=None):
        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial

    def train_step(self, dataloader):
        self.model.train()
        loss = 0.0

        size = len(dataloader.dataset)
        for batch, (user, item, label) in enumerate(dataloader):
            user = user.to(self.device)
            item = item.to(self.device)
            label = label.to(self.device)
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
        """Validation or test step"""
        # Set model to eval mode
        self.model.eval()

        loss = 0.0
        predictions, labels = [], []

        with torch.no_grad():
            for  batch, (user, item, label) in enumerate(dataloader):
                prediction = self.model(user, item)
                
                J = self.loss_fn(prediction, label).item()

                loss += (J - loss)/(batch + 1)

                # store outputs
                prediction = prediction.numpy()
                predictions.extend(prediction)
                labels.extend(label.numpy())

        return loss, np.vstack(labels), np.vstack(predictions)

    def predict_step(self, dataloader):
        """ Prediction step (inference)
        
        Loss is not calculated for this loop.
        """
        self.model.eval()
        predictions, labels = [], []

        # Interate over val batches
        with torch.no_grad():
            for batch, (user, item, label) in enumerate(dataloader):
                # Forward pass w/ inputs
                prediction = self.model(user, item)

                prediction = prediction.numpy()
                predictions.extend(prediction)
                labels.extend(label.numpy())

        return np.vstack(labels), np.vstack(predictions)

    def train(self, num_epochs, patience, train_dataloader, val_dataloader):
        best_val_loss = np.inf
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
            #mlflow.log_metrics(
            #    {'train_loss':train_loss, 'val_loss':val_loss}, step=epoch
            #)

            # Logging
            print(
                f"Epoch:  {epoch + 1} |"
                f"train_loss: {train_loss:.5f},"
                f"val_loss: {val_loss:.5f},"
                f"lr: {self.optimizer.param_groups[0]['lr']:.2E},"
                f"patience: {_patience}"
            )
        return best_val_loss, best_model

def train(
    params_fp: Path=Path(config.config_dir, "params.json"),
    #train_dataloader: torch.utils.data.DataLoader,
    #val_dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"),
    trial: optuna.trial._trial.Trial = None)->Tuple:

    params = Namespace(**utils.load_dict(params_fp))

    dataset = utils.get_data()
    n_users = dataset['user_id'].nunique() + 1
    n_items = dataset['item_id'].nunique() + 1

    # left one out validation
    dataloader = data.RCDataloader(params, dataset)
    train_dataloader = dataloader.get_train_set()
    test_dataloader = dataloader.get_test_set()

    model = models.initialize_model(
        n_users=n_users, 
        n_items=n_items, 
        params_fp=params_fp,
        device=device
    )

    loss_fn = nn.MSELoss()

    # Define optimizer & scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = "min", factor=0.05, patience=params.patience
    )

    trainer = Trainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        trial=trial
    )

    best_val_loss, best_model = trainer.train(
        params.n_epochs, params.patience, train_dataloader, test_dataloader
    )

    return params, best_model, best_val_loss
