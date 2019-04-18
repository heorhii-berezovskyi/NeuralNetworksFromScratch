import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Name import Name
from neural_nets.optimizer.Optimizer import Optimizer


class SGDNesterovMomentum(Optimizer):
    name = Name.SGD_NESTEROV_MOMENTUM
    learning_rate = 0.001
    mu = 0.9

    def __init__(self, layer_id: str, velocity_previous: Cache, velocity: Cache):
        self.id = layer_id
        self.velocity_previous = velocity_previous
        self.velocity = velocity

    @classmethod
    def init_memory(cls, layer_id: str, weights: Cache):
        velocity_previous = Cache()
        velocity = Cache()
        for name in weights.get_keys():
            w = weights.get(name=name)
            velocity_previous.add(name=name, value=np.zeros_like(w))
            velocity.add(name=name, value=np.zeros_like(w))
        return cls(layer_id=layer_id,
                   velocity_previous=velocity_previous,
                   velocity=velocity)

    def update_memory(self, layer_backward_run: Cache):
        updated_velocity_prev = Cache()
        updated_velocity = Cache()
        for name in self.velocity.get_keys():
            dw = layer_backward_run.get(name=name)
            v_prev = self.velocity.get(name)
            updated_velocity_prev.add(name=name, value=v_prev)

            v = SGDNesterovMomentum.mu * self.velocity.get(name) - SGDNesterovMomentum.learning_rate * dw
            updated_velocity.add(name=name, value=v)
        return SGDNesterovMomentum(layer_id=self.id,
                                   velocity_previous=updated_velocity_prev,
                                   velocity=updated_velocity)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.velocity.get_keys():
            v_prev = self.velocity_previous.get(name)
            v = self.velocity.get(name)

            w = weights.get(name=name)
            w += -SGDNesterovMomentum.mu * v_prev + (1. + SGDNesterovMomentum.mu) * v
            updated_weights.add(name=name, value=w)
        return updated_weights

    def memory_content(self) -> dict:
        result = {}
        optimizer_id = self.id + SGDNesterovMomentum.name.value
        for name in self.velocity.get_keys():
            velocity_param_key = optimizer_id + Name.VEL.value + name.value
            result[velocity_param_key] = self.velocity.get(name=name)

            velocity_prev_param_key = optimizer_id + Name.VEL_PREV.value + name.value
            result[velocity_prev_param_key] = self.velocity_previous.get(name=name)
        return result

    def from_params(self, all_params):
        optimizer_id = self.id + SGDNesterovMomentum.name.value

        velocity = Cache()
        velocity_previous = Cache()
        for name in self.velocity.get_keys():
            velocity_param_key = optimizer_id + Name.VEL.value + name.value
            velocity.add(name=name, value=all_params[velocity_param_key])

            velocity_prev_param_key = optimizer_id + Name.VEL_PREV.value + name.value
            velocity_previous.add(name=name, value=all_params[velocity_prev_param_key])
        return SGDNesterovMomentum(layer_id=self.id,
                                   velocity=velocity,
                                   velocity_previous=velocity_previous)
