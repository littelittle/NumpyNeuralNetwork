from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X@self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        W_grad = np.einsum('ij,ik->ijk', grad, self.input)
        if np.any(np.isnan(W_grad)):
            print(grad,self.input)
            raise ValueError("backward trouble!")
        b_grad = grad
        self.grads['W'] = np.mean(np.transpose(W_grad,(0, 2, 1)), axis=0) # average the grads along the batchsize channel
        self.grads['b'] = np.mean(b_grad, axis=0)
        passing_grad = grad@np.transpose(self.W, (1, 0)) # (batchsize, outdim)@(outdim, indim)
        return passing_grad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class Conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        self.kernel = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.inputs = None # record the input every time after forward
        self.grads = {'kernel': None, 'b': None} 
        self.params = {'kernel': self.kernel, 'b': self.b}

    def __call__(self, X:np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X:np.ndarray):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        no padding
        """
        # the 1Gen: no padding:
        batchsize, _, H, W = X.shape
        # caculate the new_H and new_W
        new_H = (H-self.kernel_size+self.padding*2)//self.stride + 1
        new_W = (W-self.kernel_size+self.padding*2)//self.stride + 1
        # initialize the output
        output = np.zeros((batchsize, self.out_channels, new_H, new_W))
        # start the for loop:
        for i in range(new_H):
            for j in range(new_W):
                # get the current window:
                # X[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] # (batchsize, in, k, k)
                # self.kernel # (out, in, k, k)

                output[:, :, i, j] = np.matmul(
                    X[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].reshape(batchsize, -1),
                    self.kernel.reshape(self.out_channels, -1).transpose(1, 0)
                ) + self.b
        self.inputs = X
        return output

    def backward(self, grads:np.ndarray):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        # for every element in the grads, compute the contributes, especially for the new_H, new_W term
        batchsize, _, new_H, new_W = grads.shape 
        passing_grad = np.zeros_like(self.inputs)
        kernel_grad = np.zeros_like(self.kernel)
        bias_grad = np.zeros_like(self.b)
        for i in range(new_H):
            for j in range(new_W):
                # add the gradients up!
                grad = grads[:, :, i, j] # (bs, out_channel) 
                kernel_grad += np.matmul(
                    grad.transpose(1, 0),   # (out_channel, bs)
                    self.inputs[:,:,i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].reshape(batchsize, -1)
                ).reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) # self.inputs is (bs, *)

                bias_grad += grad.mean(axis=0) # get the average grad along the batchsize axis

                # make sure that map the correct i, j to passing grad
                passing_grad[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += np.matmul(
                    grad, # (bs, out_channel)
                    self.kernel.reshape(self.out_channels, -1) # (out_channel, in*k*k)
                ).reshape(batchsize, self.in_channels, self.kernel_size, self.kernel_size) # (bs, inchannel, k, k)
        
        self.grads['kernel'] = kernel_grad
        self.grads['b'] = bias_grad
        return passing_grad

    def clear_grad(self):
        self.grads = {'kernel':None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        pass

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        pass
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition