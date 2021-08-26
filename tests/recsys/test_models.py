from pathlib import Path
from argparse import Namespace

import torch

from recsys import models, utils, config, data

class Testmfpt:
    """
    def setup_method(self):
        #Called before every method
        # Params, n_users and n_items are in the dataset.
        self.n_users = 944,
        self.n_items = 1683,
        self.n_factors = 20,
        self.dropout_p = 1e-3
        params = Namespace(
            n_users = self.n_users,
            n_items = self.n_items,
            n_factors = self.n_factors,
            dropout_p = self.dropout_p
        )

        # model
        utils.set_seed()
        self.mfpt = models.initialize_model(
            n_users = self.n_users,
            n_items = self.n_items
        )

    def teardown_method(self):
        #Called after every method
        del self.mfpt

    def test_initialize_model(self):
        utils.set_seed()
        model = models.mfpt(
            n_users = self.n_users,
            n_items = self.n_items,
            n_factors = self.n_factors,
            dropout_p = self.dropout_p
        )
        for params1, params2 in zip(self.model.parameters(), model.parameters()):
            assert not param1.data.ne(param2.data).sum()>0
        assert self.mfpt.n_factors == model.n_factors

    def test_init(self):
        assert self.model.user_factors.shape == (self.n_users, self.n_factors)
        assert self.model.item_factors.shape == (self.n_items, self.n_factors)
        assert self.model.users_biases.shape == (self.n_users, 1)
        assert self.model.items_biases.shape == (self.n_items, 1)

    def test_forward(self):
        user = torch.tensor([12])
        item = torch.tensor([12])
        z = self.mfpt.forward(user, item)
        assert z.shape == (len(user), len(item))
    """
    pass