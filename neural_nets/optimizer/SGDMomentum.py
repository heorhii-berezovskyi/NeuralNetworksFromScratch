import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.optimizer.Optimizer import Optimizer


class SGDMomentum(Optimizer):
    learning_rate = 0.001
    mu = 0.9

    def __init__(self, velocity: Cache):
        self.velocity = velocity

    @classmethod
    def init_memory(cls, weights: Cache):
        velocity = Cache()
        for name in weights.get_keys():
            w = weights.get(name=name)
            velocity.add(name=name, value=np.zeros_like(w))
        return cls(velocity=velocity)

    def update_memory(self, layer_backward_run: Cache):
        updated_velocity = Cache()
        for name in self.velocity.get_keys():
            dw = layer_backward_run.get(name=name)

            velocity = self.velocity.get(name=name) * SGDMomentum.mu - SGDMomentum.learning_rate * dw
            updated_velocity.add(name=name, value=velocity)
        return SGDMomentum(velocity=updated_velocity)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.velocity.get_keys():
            velocity = self.velocity.get(name=name)
            w = weights.get(name=name)
            w += velocity
            updated_weights.add(name=name, value=w)
        return updated_weights
