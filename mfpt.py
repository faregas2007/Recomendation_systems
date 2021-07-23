import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from numpy as np
from scipy.sparse import rand as sprand

import argparse import Namespace
from typing import List

def load_artifacts(run_id, device):
        """
        Load artifacts for current model
        
        Args:
            run_id (str) : ID of the model run to laod artifacts. 
            device (torch.device) : Device to run model on. Default on CPU
        """
        # add artifacts mlflow here latter
        params = Namespace(**utils.load_dict('params.json"))
        model_state = torch.load("model.pt", map_location=device)
        performance = utils.load_dict(filepath="performance.json")
        return {
            "params": params 
            "model": model
            "performance": performance
        }
    
class mfpt(nn.Module):
    def __init__(self, 
            n_users:int,
            n_items:int,
            n_factors:int,
            dropout_p: float,
            sparse: bool)->None:
        super(mfpt, self).__init__()

        self.n_users = torch.tensor(n_users, dtype=float)
        self.n_items = torch.tensor(n_items, dtype=float)
        self.n_factors = torch.tensor(n_factors, dtype=float)
        self.dropout_p = torch.tensor(dropout_p, dtype=float)
        self.sparse = sparse

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.users_biases = torch.nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = torch.nn.Embedding(n_users, 1, sparse=sparse)
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=sparse)

    def forward(self, user: int, item:int)-> torch.Tensor:
        ues = self.user_factors(user)
        uis = self.item_factors(item)

        preds = self.user_biases(user)
        preds += self.item_biases(item)

        preds += ((self.dropout(ues)*self.dropout(uis)).sum(dim=1, keepdim=True))
        return preds.squeeze()

def initialize_model(
    params: Namespace
    device: torch.device = torch.device('cpu')
)-> nn.Module:

    model = mfpt(
        n_users = int(params.n_users),
        n_items = int(params.n_items),
        n_factors = int(params.n_factors),
        dropout_p = int(params.dropout_p),
        sparse = bool(params.sparse)
    )

    model = model.to(device)

    return model
