import unittest

from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Model import TrainModel
from neural_nets.model.Trainer import Trainer
from neural_nets.optimizer.Adam import Adam


class TrainerTest(unittest.TestCase):

    def test_compile(self):
        trainer = Trainer(optimizer=Adam, loss_function=CrossEntropyLoss())
        trainer.add_linear(block_name='linear',
                           input_dim=784,
                           num_of_neurons=10)
        trainer.compile()
        self.assertTrue(len(trainer.layers) > 0, msg='Trainer cannot compile the model.')


class ModelTest(unittest.TestCase):

    def test_init_model(self):
        model = TrainModel(layers=[])
        test_model = model.to_test(model_forward_run=[])
        self.assertTrue(len(test_model.layers) == 0)
