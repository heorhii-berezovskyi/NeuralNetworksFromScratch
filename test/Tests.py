import unittest

import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Model import TrainModel
from neural_nets.model.Name import Name
from neural_nets.run.Trainer import Trainer
from neural_nets.optimizer.Adam import Adam


class TrainerTest(unittest.TestCase):

    def test_compile(self):
        trainer = Trainer(optimizer=Adam, loss_function=CrossEntropyLoss())
        trainer.add_linear(block_name='linear',
                           input_dim=784,
                           num_of_neurons=10)
        trainer.compile()
        self.assertTrue(len(trainer.layers) == 1, msg='Trainer cannot compile the model.')


class ModelTest(unittest.TestCase):

    def test_model_to_test(self):
        model = TrainModel(layers=[])
        test_model = model.to_test(model_forward_run=[])
        self.assertTrue(len(test_model.layers) == 0)

    def test_init_model(self):
        model = TrainModel(layers=[])
        model_forward_run = model.init_model()
        self.assertTrue(len(model_forward_run) == 0)


class CacheTest(unittest.TestCase):

    def test_add(self):
        cache = Cache()
        cache.add(name=Name.BETA, value=np.zeros((10, 10)))
        self.assertTrue(cache.get(name=Name.BETA).shape == np.zeros((10, 10)).shape)

    def test_get_keys(self):
        cache = Cache()
        cache.add(name=Name.BETA, value=np.zeros((10, 10)))
        cache.add(name=Name.GAMMA, value=np.zeros((10, 10)))
        self.assertIn(Name.BETA, cache.get_keys())
        self.assertIn(Name.GAMMA, cache.get_keys())

    def test_pop(self):
        cache = Cache()
        cache.add(name=Name.BETA, value=np.zeros((10, 10)))
        beta = cache.pop(name=Name.BETA)
        self.assertTrue(beta.shape == np.zeros((10, 10)).shape)

