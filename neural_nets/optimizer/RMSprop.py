import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.optimizer.Optimizer import Optimizer


class RMSprop(Optimizer):
    decay_rate = 0.99
    learning_rate = 0.001

    def __init__(self, cache: Cache, update_values: Cache):
        self.cache = cache
        self.update_values = update_values

    @classmethod
    def init_memory(cls, weights: Cache):
        cache = Cache()
        update_values = Cache()
        for name in weights.get_keys():
            w = weights.get(name=name)
            cache.add(name=name, value=np.zeros_like(w))
            update_values.add(name=name, value=np.zeros_like(w))
        return cls(cache=cache, update_values=update_values)

    def update_memory(self, layer_backward_run: Cache):
        updated_cache = Cache()
        updated_values = Cache()
        for name in self.cache.get_keys():
            dw = layer_backward_run.get(name=name)

            cache = RMSprop.decay_rate * self.cache.get(name=name) + (1. - RMSprop.decay_rate) * dw ** 2
            updated_cache.add(name=name, value=cache)

            value = -RMSprop.learning_rate * dw / (np.sqrt(cache) + 1e-6)
            updated_values.add(name=name, value=value)
        return RMSprop(cache=updated_cache, update_values=updated_values)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.cache.get_keys():
            update_values = self.update_values.get(name=name)
            w = weights.get(name=name)
            w += update_values
            updated_weights.add(name=name, value=w)
        return updated_weights
