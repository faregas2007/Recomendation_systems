from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy.sparse import rand as sprand

from argparse import Namespace
from typing import List

#from utils import *
#from config import *

from recsys import utils, config

class mfpt(nn.Module):
    def __init__(self, 
            n_users: int,
            n_items: int,
            n_factors: int,
            dropout_p: float)->None:
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

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.users_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1, sparse=True)
        self.items_biases = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=1, sparse=True)
        self.user_factors = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_factors, sparse=True)

    def forward(self, user: int, item:int)-> torch.Tensor:
        """
        Forward pass through the model. For a single user and item, this
        looks like:


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
        
        user_embedding = self.user_factors(user).float()
        item_embedding = self.item_factors(item).float()

        preds = self.users_biases(user).float()
        preds += self.items_biases(item).float()

        preds += torch.mul(self.dropout(user_embedding), self.dropout(item_embedding)).sum(dim=1, keepdim=True)
        #preds += ((self.dropout(user_embedding)*self.dropout(item_embedding)).sum(dim=1, keepdim=True))
        return preds.reshape(-1).squeeze()
    
    def get_factors(self, users, items):
        user_embedding = self.user_factors(users).float()
        item_embedding = self.item_factors(items).float()
                                             
        return user_embedding, item_embedding
                                             
    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)


def initialize_model(
    n_users: int=utils.get_data()["user_id"].nunique() + 1,
    n_items: int=utils.get_data()["item_id"].nunique() + 1,
    params_fp: Path = Path(config.config_dir, "params.json"),
    device: torch.device = torch.device('cpu')
    )-> nn.Module:

    params = Namespace(**utils.load_dict(params_fp))                               
                                             
    #dataset = utils.get_data()
    #n_users = dataset['user_id'].nunique() + 1
    #n_items = dataset['item_id'].nunique() + 1
                                             
    model = mfpt(
        n_users = n_users,
        n_items = n_items,
        n_factors = params.n_factors,
        dropout_p = params.dropout_p
    )

    model = model.to(device)

    return model
