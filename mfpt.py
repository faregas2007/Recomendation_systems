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
        return {
            "params": params 
        }

class mfpt(nn.Module):
    def __init__(self, 
            n_users: int,
            n_items: int,
            n_factors: int,
            dropout_p: float,
            sparse: bool) -> None:
        """
        Parameters
        ----------
        n_users: int
            Number of users
        n_items: int
            Number of items
        n_factors: int
            Number of latent factors (or embeddings or whatever you want to call it)
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse: bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use weight decay
            on the optimizer if sparse=True. Also, can only use Adagrad.
        """
        super(mfpt, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout_p = dropout_p
        self.sparse = sparse

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.users_biases = torch.nn.Embedding(self.n_users, 1, sparse=self.sparse)
        self.item_biases = torch.nn.Embedding(self.n_users, 1, sparse=self.sparse)
        self.user_factors = torch.nn.Embedding(self.n_users, self.n_factors, sparse=self.sparse)
        self.item_factors = torch.nn.Embedding(self.n_items, self.n_factors, sparse=self.sparse)

    def forward(self, user: int, item:int)-> torch.Tensor:
        """
        Forward pass through the model. For a single user and item, this
        looks like:

        user_bias + item_bias + user_embeddings.dot(item_embeddings)

        Parameters
        ----------
        users: np.ndarray
            Array of user indices
        items: np.ndarray
            Array of item indices

        Returns 
        -------
        preds: np.ndarray
            Predicted ratings
        """
        
        ues = self.user_factors(user)
        uis = self.item_factors(item)

        preds = self.user_biases(user)
        preds += self.item_biases(item)

        preds += ((self.dropout(ues)*self.dropout(uis)).sum(dim=1, keepdim=True))
        return preds.squeeze()

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)


def initialize_model(
    params: Namespace,
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
