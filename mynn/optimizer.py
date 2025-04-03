from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]  # A VERY IMPORTANT DIFFERENCE!!! * -= and * = * -
                    # print(f"the num of nonezero grad of {key} is {layer.grads[key][layer.grads[key]!=0].shape}")
                    if (key == "W") and np.any(layer.params[key] != layer.W):
                        print("self.params['key'] is not equal to self.key!!!")


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        pass
    
    def step(self):
        pass