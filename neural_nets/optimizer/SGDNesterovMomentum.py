import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.optimizer.Optimizer import Optimizer


class SGDNesterovMomentum(Optimizer):
    learning_rate = 0.001
    mu = 0.9

    def __init__(self, velocity_prev: Cache, velocity: Cache):
        self.velocity_previous = velocity_prev
        self.velocity = velocity

    @classmethod
    def init_memory(cls, weights: Cache):
        velocity_prev = Cache()
        velocity = Cache()
        for name in weights.get_keys():
            w = weights.get(name=name)
            velocity_prev.add(name=name, value=np.zeros_like(w))
            velocity.add(name=name, value=np.zeros_like(w))
        return cls(velocity_prev=velocity_prev, velocity=velocity)

    def update_memory(self, layer_backward_run: Cache):
        updated_velocity_prev = Cache()
        updated_velocity = Cache()
        for name in self.velocity.get_keys():
            dw = layer_backward_run.get(name=name)
            v_prev = self.velocity.get(name)
            updated_velocity_prev.add(name=name, value=v_prev)

            v = SGDNesterovMomentum.mu * self.velocity.get(name) - SGDNesterovMomentum.learning_rate * dw
            updated_velocity.add(name=name, value=v)
        return SGDNesterovMomentum(velocity_prev=updated_velocity_prev, velocity=updated_velocity)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.velocity.get_keys():
            v_prev = self.velocity_previous.get(name)
            v = self.velocity.get(name)

            w = weights.get(name=name)
            w += -SGDNesterovMomentum.mu * v_prev + (1. + SGDNesterovMomentum.mu) * v
            updated_weights.add(name=name, value=w)
        return updated_weights
