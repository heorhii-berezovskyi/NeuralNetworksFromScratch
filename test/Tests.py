import unittest

from numpy.random import randint
from numpy.random import randn

from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Linear import Linear
from neural_nets.model.Model import Model


class ModelTest(unittest.TestCase):

    def test_add_layer(self):
        batch_size = 64
        model = Model(reg=0.01)
        linear = Linear(10, (784, batch_size))
        model.add_layer(linear)
        self.assertIn(linear, model.layers)

    def test_forward(self):
        batch_size = 64
        model = Model(reg=0.01)
        linear = Linear(10, (784, batch_size))
        cross_entropy_loss = CrossEntropyLoss((10, 64))

        model.add_layer(linear)
        model.add_layer(cross_entropy_loss)

        images = randn(64, 784)
        labels = randint(0, 10, size=batch_size)
        loss = model.forward(labels=labels, images=images)

        self.assertGreater(loss, 0.0, "Cannot be zero or less than zeros")
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)

    def test_backward(self):
        batch_size = 64
        model = Model(reg=0.01)
        linear = Linear(10, (784, batch_size))
        cross_entropy_loss = CrossEntropyLoss((10, 64))

        model.add_layer(linear)
        model.add_layer(cross_entropy_loss)

        images = randn(64, 784)
        labels = randint(0, 10, size=batch_size)
        model.forward(labels=labels, images=images)
        model.backward_on_model()
        self.assertTrue(len(model.gradients) == 2, "Backpropogation doesn't work.")
