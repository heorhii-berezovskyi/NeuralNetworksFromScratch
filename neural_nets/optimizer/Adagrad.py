import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Name import Name
from neural_nets.optimizer.Optimizer import Optimizer


class Adagrad(Optimizer):
    name = Name.ADAGRAD
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

            cache = self.cache.get(name=name) + dw ** 2
            updated_cache.add(name=name, value=cache)

            value = -Adagrad.learning_rate * dw / (np.sqrt(cache) + 1e-6)
            updated_values.add(name=name, value=value)
        return Adagrad(layer_id=self.id,
                       cache=updated_cache,
                       update_values=updated_values)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.cache.get_keys():
            update_values = self.update_values.get(name=name)
            w = weights.get(name=name)
            w += update_values
            updated_weights.add(name=name, value=w)
        return updated_weights

    def memory_content(self) -> dict:
        result = {}
        optimizer_id = self.id + Adagrad.name.value
        for name in self.cache.get_keys():
            item_key = optimizer_id + name.value
            result[item_key] = self.cache.get(name=name)
        return result

    def from_params(self, all_params):
        cache = Cache()
        optimizer_id = self.id + Adagrad.name.value
        for cache_name in self.cache.get_keys():
            cache_key = optimizer_id + cache_name.value
            cache.add(name=cache_name, value=all_params[cache_key])
        return Adagrad(layer_id=self.id,
                       cache=cache,
                       update_values=self.update_values)
