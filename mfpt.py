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

def Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix
    Users are rows
    Items are columns
    Elements of the matrix are the ratings given by a user for an item
    """
    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]
    
    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz

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
