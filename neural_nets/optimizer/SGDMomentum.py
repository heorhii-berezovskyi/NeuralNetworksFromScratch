import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Name import Name
from neural_nets.optimizer.Optimizer import Optimizer


class SGDMomentum(Optimizer):
    name = Name.SGD_MOMENTUM
    learning_rate = 0.001
    mu = 0.9

    def __init__(self, layer_id: str, velocity: Cache):
        self.id = layer_id
        self.velocity = velocity

    @classmethod
    def init_memory(cls, layer_id: str, weights: Cache):
        velocity = Cache()
        for name in weights.get_keys():
            w = weights.get(name=name)
            velocity.add(name=name, value=np.zeros_like(w))
        return cls(layer_id=layer_id,
                   velocity=velocity)

    def update_memory(self, layer_backward_run: Cache):
        updated_velocity = Cache()
        for name in self.velocity.get_keys():
            dw = layer_backward_run.get(name=name)

            velocity = self.velocity.get(name=name) * SGDMomentum.mu - SGDMomentum.learning_rate * dw
            updated_velocity.add(name=name, value=velocity)
        return SGDMomentum(layer_id=self.id,
                           velocity=updated_velocity)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.velocity.get_keys():
            velocity = self.velocity.get(name=name)
            w = weights.get(name=name)
            w += velocity
            updated_weights.add(name=name, value=w)
        return updated_weights

    def memory_content(self) -> dict:
        result = {}
        optimizer_id = self.id + SGDMomentum.name.value
        for name in self.velocity.get_keys():
            velocity_item_key = optimizer_id + name.value
            result[velocity_item_key] = self.velocity.get(name=name)

        learning_rate_key = optimizer_id + Name.LEARNING_RATE.value
        result[learning_rate_key] = SGDMomentum.learning_rate

        mu_key = optimizer_id + Name.MU.value
        result[mu_key] = SGDMomentum.mu
        return result

    def from_params(self, all_params):
        velocity = Cache()
        optimizer_id = self.id + SGDMomentum.name.value
        for name in self.velocity.get_keys():
            velocity_item_key = optimizer_id + name.value
            velocity.add(name=name, value=all_params[velocity_item_key])

        learning_rate_key = optimizer_id + Name.LEARNING_RATE.value
        SGDMomentum.learning_rate = all_params[learning_rate_key]

        mu_key = optimizer_id + Name.MU.value
        SGDMomentum.mu = all_params[mu_key]
        return SGDMomentum(layer_id=self.id,
                           velocity=velocity)
