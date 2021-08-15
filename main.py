import json
from argparse import Namespace
from pathlib import Path
from typing import Dict

import mlflow
import optuna
import numpy as np
import torch

from models import *
from train import *
from eval import *
from data import *
from utils import *

def objective(
    params_fp: Path,
    device: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    trial: optuna.trial._trial.Trial)->float:

    params = Namespace(**utils.load_dict(params_fp))

    params.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 1e-5, 1e-4)
    params.threshold = trial.suggest_uniform("threshold", 3.5, 4)

    artifacts = train_model(params_fp=params_fp, device=device, trial=trial)

    params = artifacts['params']
    performance = artifacts['performance']

    trial.set_user_attr("f1", performance["overall"]["f1"])
    trial.set_user_attr("recall", performance["overall"]["recall"])
    trial.set_user_attr("precision", performance["overall"]["precision"])
    trial.set_user_attr("HR", performance["overall"]["HR"])
    trial.set_user_attr("NDCG", performance["overall"]["NDCG"])

    if params.save_model:
        torch.save(artifacts["model"].state_dict(), params.model+"recsys.pkl")

    return performance
 
def train_model(
    params_fp: Path,
    device: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    trial: optuna.trial._trial.Trial=None
)->Dict:
    """Operations for training

    Args:
        params (Namespace): Input parameters for operations
        trial (optuna.trial._trial.Trial, optional): Optuna optimization trial

    Returns:
        Artifact to save and loads
    """

    # params
    params = Namespace(**utils.load_dict(params_fp))

    # Intiialze model
    dataset = utils.get_data()

    n_users = dataset['user_id'].nunique() + 1
    n_items = dataset['item_id'].nunique() + 1

    dataloader = data.RCDataloader(params, dataset)
    train_dataloader = dataloader.get_train_set()
    test_dataloader = dataloader.get_test_set()

    model = models.initialize_model(
        params_fp = params_fp,
        device = device,
        n_users = n_users, 
        n_items = n_items)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    trainer = train.Trainer(model=model, device=device, loss_fn=loss_func, optimizer=optimizer, scheduler=scheduler)
    best_val_loss, best_model =  trainer.train(
        params.n_epochs,
        params.patience,
        train_dataloader,
        test_dataloader
    )
    
    artifacts = {
        "params": params,
        "model": best_model,
        "loss":best_val_loss,
    }

    device = torch.device("cpu")
    performance = evaluate(
        params_fp=params_fp,
        model = best_model,
        dataloader=test_dataloader,
        device=device
        )
    
    artifacts['performance'] = performance

    return artifacts
