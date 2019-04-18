import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Name import Name
from neural_nets.optimizer.Optimizer import Optimizer


class RMSprop(Optimizer):
    name = Name.RMSPROP
    decay_rate = 0.99
    learning_rate = 0.001

    def __init__(self, layer_id: str, cache: Cache, update_values: Cache):
        self.id = layer_id
        self.cache = cache
        self.update_values = update_values

    @classmethod
    def init_memory(cls, layer_id: str, weights: Cache):
        cache = Cache()
        update_values = Cache()
        for name in weights.get_keys():
            w = weights.get(name=name)
            cache.add(name=name, value=np.zeros_like(w))
            update_values.add(name=name, value=np.zeros_like(w))
        return cls(layer_id=layer_id,
                   cache=cache,
                   update_values=update_values)

    def update_memory(self, layer_backward_run: Cache):
        updated_cache = Cache()
        updated_values = Cache()
        for name in self.cache.get_keys():
            dw = layer_backward_run.get(name=name)

            cache = RMSprop.decay_rate * self.cache.get(name=name) + (1. - RMSprop.decay_rate) * dw ** 2
            updated_cache.add(name=name, value=cache)

            value = -RMSprop.learning_rate * dw / (np.sqrt(cache) + 1e-6)
            updated_values.add(name=name, value=value)
        return RMSprop(layer_id=self.id,
                       cache=updated_cache,
                       update_values=updated_values)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.cache.get_keys():
            w = weights.get(name=name)
            w += self.update_values.get(name=name)
            updated_weights.add(name=name, value=w)
        return updated_weights

    def memory_content(self) -> dict:
        result = {}
        optimizer_id = self.id + RMSprop.name.value
        for name in self.cache.get_keys():
            cache_item_key = optimizer_id + name.value
            result[cache_item_key] = self.cache.get(name=name)

        learning_rate_key = optimizer_id + Name.LEARNING_RATE.value
        result[learning_rate_key] = RMSprop.learning_rate

        decay_rate_key = optimizer_id + Name.DECAY_RATE.value
        result[decay_rate_key] = RMSprop.decay_rate
        return result

    def from_params(self, all_params):
        optimizer_id = self.id + RMSprop.name.value

        cache = Cache()
        for name in self.cache.get_keys():
            cache_item_key = optimizer_id + name.value
            cache.add(name=name, value=all_params[cache_item_key])

        learning_rate_key = optimizer_id + Name.LEARNING_RATE.value
        RMSprop.learning_rate = all_params[learning_rate_key]

        decay_rate_key = optimizer_id + Name.DECAY_RATE.value
        RMSprop.decay_rate = all_params[decay_rate_key]
        return RMSprop(layer_id=self.id,
                       cache=cache,
                       update_values=self.update_values)
