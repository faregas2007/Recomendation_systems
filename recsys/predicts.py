from distutils.util import strtobool
from typing import Dict, List

import numpy as np
import torch

from recsys import data, train, utils, eval

def predict(
    artifacts: Dict,
    dataset: torch.utils.data.Dataset = utils.get_data(),
    device: torch.device = torch.device('cpu')
    )->Dict:
    """Predict ratings for an input params, using
    best model from the "best" experiment.
    
    Args:
        artifacts: call from the best experiment
        dataloader: inference/production/feature_store dataloader
        device: device

    Return:
        Performance, predicted ratings
    """

    params = artifacts["params"]
    model = artifacts["model"]
    
    # the dataset need to be change to batch data stored inside feature_store.
    # dataloader --> need to have a create_dataloader, instead of train and test dataloader --> to do list.
    # for inference --> to do list. 
    dataloader = data.RCDataloader(params, dataset)
    dataloader = dataloader.get_test_set()

    # Intiialze model
    trainer = train.Trainer(model, device)
    y_true, y_pred = trainer.predict_step(dataloader=dataloader)

    y_true = eval.binary_feedback(y_true, params.threshold)
    y_pred = eval.binary_feedback(y_pred, params.threshold)
    performance = {}
    performance = eval.get_metrics(model, dataloader, params.top_k, y_true, y_pred, device) 

    return y_true, y_pred, performance
    
def item_recommendations(
    item_id: int,
    top_k: int, 
    artifacts: Dict
    )->Dict:
    """item-item recommendation. Top-k based on predicted ratings

    Args:
        item_id: item_id of the movie
        best_model: the best_model from best_experiment artifact
        dataloader: dataloader for inference.
    
    Return:
        return topk recommened items based on predicted ratings.
    """
    
    best_model = artifacts["model"]
    params = artifacts["params"]

    dataset = utils.get_data()

    # the dataset need to be change data stored inside database.
    # dataloader --> need to have a create_dataloader, instead of train and test dataloader --> to do list.
    # for inference --> To do list. 
    dataloader = data.RCDataloader(params, dataset)
    dataloader = dataloader.get_test_set()

    items_id = []
    users_id = []
    for user, item, _ in dataloader:
        items_id.append(item)
        users_id.append(user)

    items_id = torch.cat(items_id)
    users_id = torch.cat(users_id)

    item_id_index = (items_id==item_id).nonzero(as_tuple=False)
    user_id = users_id[item_id_index]

    dataset = utils.get_data()
    items = torch.tensor(dataset['item_id'])

    predictions = best_model(users_id, torch.tensor([item_id]))
    _, indices = torch.topk(predictions, top_k)
    recommends = torch.take(items, indices)

    return (dataset['title'][recommends.cpu().detach().numpy()])