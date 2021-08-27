from pathlib import Path
from argparse import Namespace

import torch

from recsys import models, utils, config, data

class Testmfpt:
    
    def setup_method(self):
        #Called before every method
        # Params, n_users and n_items are in the dataset.
        self.n_users = 944
        self.n_items = 1683
        self.n_factors = 20
        self.dropout_p = 1e-3
        self.device = torch.device("cpu")
        self.params_fp = "/users/tp/dat/Recomendation_systems/config/parmas.json"
        self.user = torch.tensor([1,2])
        self.item = torch.tensor([1,2])
        params = Namespace(
            n_users = self.n_users,
            n_items = self.n_items,
            n_factors = self.n_factors,
            dropout_p = self.dropout_p
        )

        # model
        utils.set_seed()
        self.mfpt = models.initialize_model(
            n_users = int(self.n_users),
            n_items = int(self.n_items),
            params_fp = str(Path(config.config_dir, "params.json")),
            device = torch.device("cpu")
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
        for params1, params2 in zip(model.parameters(), self.mfpt.parameters()):
            assert not params1.data.ne(params2.data).sum()>0
        assert self.mfpt.n_factors == model.n_factors

    def test_init(self):
        assert tuple(self.mfpt.user_factors(self.user).shape) == (self.user.shape[0], self.n_factors)
        assert tuple(self.mfpt.item_factors(self.item).shape) == (self.item.shape[0], self.n_factors)
        assert tuple(self.mfpt.users_biases(self.user).shape) == (self.user.shape[0], 1)
        assert tuple(self.mfpt.items_biases(self.item).shape) == (self.item.shape[0], 1)

    def test_forward(self):
        assert len(self.user) == len(self.item)
        z = self.mfpt.forward(self.user, self.item)
        assert len(z) == 2

    def test_get_factors(self):
        user_embedding, item_embedding = self.mfpt.get_factors(self.user, self.item)
        assert tuple(user_embedding.shape) == (len(self.user), self.n_factors) 
        assert tuple(item_embedding.shape) == (len(self.item), self.n_factors)