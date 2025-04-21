from .op import *
from .init_method import *
import pickle
import sys
from .time_detector import *

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=Xavier_init)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)
    
    def __str__(self):
        return f"""A MLP Model With Whose Sublayer is as below:
                    {[layer.__str__()  for layer in self.layers]}
                """

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        # import ipdb; ipdb.set_trace()
        try:
            for i, layer in enumerate(reversed(self.layers)):
                grads = layer.backward(grads)
        except ValueError:
            print(f"there is nan in layer{i}\nwhose in/out size is {layer.W.shape}!")
            sys.exit(0)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, debug=False):
        self.size_list = size_list
        self.act_func = act_func
        self.lambda_list = lambda_list.copy() if lambda_list is not None else None

        if size_list is not None and act_func is not None:
            self.layers = []
            for i, size in enumerate(size_list):
                append_activate = False
                if size == "reshape":
                    layer = Reshape()
                    append_activate = False
                if len(size)==3:
                    layer = Conv2D(in_channels=size[0], out_channels=size[1], kernel_size=size[2], initialize_method=Xavier_init)
                    append_activate = False
                elif len(size)==2:
                    layer = Linear(in_dim=size[0], out_dim=size[1], initialize_method=Xavier_init)
                    append_activate = True
                elif len(size)==1:
                    # in this case, it is just the max pool layer specified with the kernel size
                    layer = MaxPool(kernel_size=size[0])
                    append_activate = True
                if lambda_list is not None and layer.optimizable:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list.pop(0)
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 1 and append_activate:
                    self.layers.append(layer_f)

        if debug:
            for layer in self.layers:
                layer.forward = timing_decorator(layer.forward)
                layer.backward = timing_decorator(layer.backward)

    def __call__(self, X):
        return self.forward(X)
    
    def __str__(self):
        return f"""A CNN Model With Whose Sublayer is as below:
                    {[layer.__str__()  for layer in self.layers]}
                """

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X.reshape(-1, 1, 28, 28)
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        # import ipdb; ipdb.set_trace()
        try:
            for i, layer in enumerate(reversed(self.layers)):
                grads = layer.backward(grads)
        except ValueError:
            print(f"there is nan in layer{i}\nwhose in/out size is {layer.W.shape}!")
            sys.exit(0)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.lambda_list = param_list[2]
        self.dev_score = param_list[3]
        i = 1
        for layer in self.layers:
            if layer.optimizable:
                i += 1
                if layer.__class__.__name__ == "Linear":
                    layer.W = param_list[i + 2]['W']
                    layer.b = param_list[i + 2]['b']
                    layer.params['W'] = layer.W
                    layer.params['b'] = layer.b
                    layer.weight_decay = param_list[i + 2]['weight_decay']
                    layer.weight_decay_lambda = param_list[i+2]['lambda']
                elif layer.__class__.__name__ == "Conv2D":
                    layer.kernel = param_list[i + 2]['kernel']
                    layer.b = param_list[i + 2]['b']
                    layer.params['kernel'] = layer.kernel
                    layer.params['b'] = layer.b
                    layer.weight_decay = param_list[i + 2]['weight_decay']
                    layer.weight_decay_lambda = param_list[i+2]['lambda']
        
        
    def save_model(self, save_path, dev_score=0):
        param_list = [self.size_list, self.act_func, self.lambda_list, dev_score]
        for layer in self.layers:
            if layer.optimizable:
                if layer.__class__.__name__ == "Linear":
                    param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
                elif layer.__class__.__name__ == "Conv2D":
                    param_list.append({'kernel' : layer.params['kernel'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)