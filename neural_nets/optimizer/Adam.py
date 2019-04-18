import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Name import Name
from neural_nets.optimizer.Optimizer import Optimizer


class Adam(Optimizer):
    name = Name.ADAM
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999

    def __init__(self, layer_id: str, num_iter: int, first_moment: Cache, second_moment: Cache, update_values: Cache):
        self.id = layer_id
        self.num_iter = num_iter
        self.first_moment = first_moment
        self.second_moment = second_moment
        self.update_values = update_values

    @classmethod
    def init_memory(cls, layer_id: str, weights: Cache):
        first_moment = Cache()
        second_moment = Cache()
        update_values = Cache()
        for name in weights.get_keys():
            w = weights.get(name=name)
            first_moment.add(name=name, value=np.zeros_like(w))
            second_moment.add(name=name, value=np.zeros_like(w))
            update_values.add(name=name, value=np.zeros_like(w))
        return cls(layer_id=layer_id,
                   num_iter=1,
                   first_moment=first_moment,
                   second_moment=second_moment,
                   update_values=update_values)

    def update_memory(self, layer_backward_run: Cache):
        updated_first_moment = Cache()
        updated_second_moment = Cache()
        updated_values = Cache()
        for name in self.first_moment.get_keys():
            dw = layer_backward_run.get(name=name)

            f_m = Adam.beta1 * self.first_moment.get(name=name) + (1. - Adam.beta1) * dw
            updated_first_moment.add(name=name, value=f_m)
            f_m_t = f_m / (1. - Adam.beta1 ** self.num_iter)

            s_m = Adam.beta2 * self.second_moment.get(name=name) + (1. - Adam.beta2) * (dw ** 2)
            updated_second_moment.add(name=name, value=s_m)
            s_m_t = s_m / (1. - Adam.beta2 ** self.num_iter)

            value = -Adam.learning_rate * f_m_t / (np.sqrt(s_m_t) + 1e-6)
            updated_values.add(name=name, value=value)
        return Adam(layer_id=self.id,
                    num_iter=self.num_iter + 1,
                    first_moment=updated_first_moment,
                    second_moment=updated_second_moment,
                    update_values=updated_values)

    def update_weights(self, weights: Cache) -> Cache:
        updated_weights = Cache()
        for name in self.first_moment.get_keys():
            update_values = self.update_values.get(name=name)
            w = weights.get(name=name)
            w += update_values
            updated_weights.add(name=name, value=w)
        return updated_weights

    def memory_content(self) -> dict:
        result = {}
        optimizer_id = self.id + Adam.name.value
        for name in self.first_moment.get_keys():
            first_moment_param_key = optimizer_id + Name.FIRST_MOM.value + name.value
            result[first_moment_param_key] = self.first_moment.get(name=name)

            second_moment_param_key = optimizer_id + Name.SECOND_MOM.value + name.value
            result[second_moment_param_key] = self.second_moment.get(name=name)
        num_iter_key = optimizer_id + Name.NUM_ITER.value
        result[num_iter_key] = np.array([self.num_iter], dtype=int)
        return result

    def from_params(self, all_params):
        optimizer_id = self.id + Adam.name.value

        first_moment = Cache()
        second_moment = Cache()

        for name in self.first_moment.get_keys():
            first_moment_param_key = optimizer_id + Name.FIRST_MOM.value + name.value
            first_moment.add(name=name, value=all_params[first_moment_param_key])

            second_moment_param_key = optimizer_id + Name.SECOND_MOM.value + name.value
            second_moment.add(name=name, value=all_params[second_moment_param_key])

        num_iter_key = optimizer_id + Name.NUM_ITER.value
        num_iter_value = all_params[num_iter_key][0]
        return Adam(layer_id=self.id,
                    num_iter=num_iter_value,
                    first_moment=first_moment,
                    second_moment=second_moment,
                    update_values=self.update_values)
