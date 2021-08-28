from pathlib import Path
from argparse import Namespace
from typing import Dict, List, Tuple
import numpy as np

import torch

from sklearn.metrics import precision_recall_fscore_support
#from data import *
#from train import *
#from predict import *
from recsys import train, data, config, utils



def hit(ng_item, pred_item):
    if ng_item in pred_item:
        return 1
    return 0

def ndcg(ng_item, pred_item):
    if ng_item in pred_item:
        index = pred_item.index(ng_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def rec_metrics(model,test_loader, top_k, device):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recomends = torch.take(item, indices).cpu().numpy().tolist()

        ng_item = item[0].item()
        HR.append(hit(ng_item, recomends))
        NDCG.append(ndcg(ng_item, recomends))

    return np.mean(HR), np.mean(NDCG)


def binary_feedback(ratings, threshold):
    ratings = np.array(ratings).flatten()
    #mean_ratings = ratings.mean()
    # normalized ratings
    normalize = ratings - threshold
    normalize2 = []

    # taken ratings as the binary feedback.
    # based on mean
    # todo: improvement based on k-threshold evaluation. 
    for i in range(len(ratings)):
        normalize2.append(0 if normalize[i] < 0 else 1)

    return normalize2

def get_metrics(
    model,
    dataloader: torch.utils.data.DataLoader,
    top_k: int, # have to derived from the threshold, initial value from params.json.
    y_true: np.ndarray,
    y_pred: np.ndarray,
    device: torch.device("cuda:0" if torch.cuda.is_available else "cpu")
)->Dict:
    """Performance metrics

    Args:
        HR: Hit Rated
        NDCG: Normalized Discounted Cumulative Gain
        precision: precision metrics, to determine threshold and top_k value
        recall: recall metrics, to determine threshold and top_k value
        f1: hamornic average of precision and recall.
    
    Returns:
        Dictionary of overall performance metrics
    """

    metrics = {"overall":{}}
    HR, NDCG = rec_metrics(model, dataloader, top_k, device)

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    
    # dependence on the top_k, derived metrics
    metrics["overall"]["HR"] = HR
    metrics["overall"]["NDCG"] = NDCG

    return metrics

def evaluate(
    model,
    dataloader: torch.utils.data.DataLoader,
    params_fp: Path = Path(config.config_dir, "params.json"),
    device: torch.device = torch.device("cpu"),
    #device: torch.device("cuda:0" if torch.cuda.is_available else "cpu"),
    #trial: None,

    #artifacts: Dict,
)->Tuple:
    """Evaluate performance on data.

    Args:
        model: should be the best_model after the training step
        dataloader (torch.utils.data.Dataloader): any dataloader set
        device (torch.device): Device to run model on. Defaults on CPU

    Returns:
        Grouth truth ratings and predicted labels/ratings, performance/other metrics
    """
    params = Namespace(**utils.load_dict(params_fp))
    
    metrics = {"overall":{}}


    trainer = train.Trainer(model, device)
    y_true, y_pred = trainer.predict_step(dataloader=dataloader)

    y_true = binary_feedback(y_true, params.threshold)
    y_pred = binary_feedback(y_pred, params.threshold)
    performance = {}
    performance = get_metrics(model, dataloader, params.top_k, y_true, y_pred, device) 

    return performance




