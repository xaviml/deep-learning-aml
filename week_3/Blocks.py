from __future__ import print_function, absolute_import, division #You don't need to know what this is. 
import numpy as np #this imports numpy, which is used for vector- and matrix calculations

def dense_forward(x_input, W, b):
    """Perform the mapping of the input
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the output of a dense layer 
        np.array of size `(n_objects, b_out)`
    """
    #################
    ### YOUR CODE ###
    #################
    output = np.add(np.dot(x_input, W), b)
    return output

def dense_grad_input(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to the input of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dense layer 
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss with 
        respect to the input of the layer
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    grad_input = np.dot(grad_output, W.T) # The partial derivate of H (XW + b) respect X is W.T
    return grad_input

def dense_grad_W(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to W parameter of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dense layer 
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss 
        with respect to W parameter of the layer
        np.array of size `(n_in, n_out)`
    """
    #################
    ### YOUR CODE ###
    #################
    grad_W = np.dot(x_input.T, grad_output)
    return grad_W

def dense_grad_b(x_input, grad_output, W, b):
    """Calculate the partial derivative of 
        the loss with respect to b parameter of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dense layer 
            np.array of size `(n_objects, n_out)`
        W: np.array of size `(n_in, n_out)`
        b: np.array of size `(n_out,)`
    # Output
        the partial derivative of the loss 
        with respect to b parameter of the layer
        np.array of size `(n_out,)`
    """
    #################
    ### YOUR CODE ###
    #################
    grad_b = np.dot(grad_output.T, np.ones(grad_output.shape[0]))
    return grad_b

class Layer(object):
    
    def __init__(self):
        self.training_phase = True
        self.output = 0.0
        
    def forward(self, x_input):
        self.output = x_input
        return self.output
    
    def backward(self, x_input, grad_output):
        return grad_output
    
    def get_params(self):
        return []
    
    def get_params_gradients(self):
        return []
    

class Dense(Layer):
    
    def __init__(self, n_input, n_output):
        super(Dense, self).__init__()
        #Randomly initializing the weights from normal distribution
        self.W = np.random.normal(size=(n_input, n_output))
        self.grad_W = np.zeros_like(self.W)
        #initializing the bias with zero
        self.b = np.zeros(n_output)
        self.grad_b = np.zeros_like(self.b)
      
    def forward(self, x_input):
        self.output = dense_forward(x_input, self.W, self.b)
        return self.output
    
    def backward(self, x_input, grad_output):
        # get gradients of weights
        self.grad_W = dense_grad_W(x_input, grad_output, self.W, self.b)
        self.grad_b = dense_grad_b(x_input, grad_output, self.W, self.b)
        # propagate the gradient backwards
        return dense_grad_input(x_input, grad_output, self.W, self.b)
    
    def get_params(self):
        return [self.W, self.b]

    def get_params_gradients(self):
        return [self.grad_W, self.grad_b]

def relu_forward(x_input):
    """relu nonlinearity
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
    # Output
        the output of relu layer
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    output = np.maximum(0,x_input)
    return output

def relu_grad_input(x_input, grad_output):
    """relu nonlinearity gradient. 
        Calculate the partial derivative of the loss 
        with respect to the input of the layer
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
        grad_output: np.array of size `(n_objects, n_in)`
    # Output
        the partial derivative of the loss 
        with respect to the input of the layer
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    
    partial_derivate = x_input
    partial_derivate[partial_derivate <= 0] = 0
    partial_derivate[partial_derivate > 0] = 1
    grad_input = grad_output * partial_derivate
    return grad_input

class ReLU(Layer):
        
    def forward(self, x_input):
        self.output = relu_forward(x_input)
        return self.output
    
    def backward(self, x_input, grad_output):
        return relu_grad_input(x_input, grad_output)

class SequentialNN(object):

    def __init__(self):
        self.layers = []
        self.training_phase = True
        
    def set_training_phase(self, is_training=True):
        self.training_phase = is_training
        for layer in self.layers:
            layer.training_phase = is_training
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, x_input):
        self.output = x_input
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output
    
    def backward(self, x_input, grad_output):
        inputs = [x_input] + [l.output for l in self.layers[:-1]]
        for input_, layer_ in zip(inputs[::-1], self.layers[::-1]):
            grad_output = layer_.backward(input_, grad_output)
            
    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
    
    def get_params_gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_params_gradients())
        return grads
    
class Loss(object):
    
    def __init__(self):
        self.output = 0.0
        
    def forward(self, target_pred, target_true):
        return self.output
    
    def backward(self, target_pred, target_true):
        return np.zeros_like(target_pred)

def hinge_forward(target_pred, target_true):
    """Compute the value of Hinge loss 
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the value of Hinge loss 
        for a given prediction and the ground truth
        scalar
    """
    #################
    ### YOUR CODE ###
    #################
    n = len(target_true)
    output = 0

    for pred, true in zip(target_pred, target_true):
        output += np.maximum(0, 1 - np.dot(pred,true))
    output /= n
    return output

def hinge_grad_input(target_pred, target_true):
    """Compute the partial derivative 
        of Hinge loss with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the partial derivative 
        of Hinge loss with respect to its input
        np.array of size `(n_objects,)`
    """

    #################
    ### YOUR CODE ###
    #################

    n = len(target_true)
    output = np.zeros(target_true.shape)
    for i, val in enumerate(target_true):
        p = np.dot(target_pred[i], val)
        if p < 1:
            output[i] = -val/n
    grad_input = output
    
    return grad_input

class Hinge(Loss):
    
    def forward(self, target_pred, target_true):
        self.output = hinge_forward(target_pred, target_true)
        return self.output
    
    def backward(self, target_pred, target_true):
        return hinge_grad_input(target_pred, target_true)
    
def l2_regularizer(weight_decay, weights):
    """Compute the L2 regularization term
    # Arguments
        weight_decay: float
        weights: list of arrays of different shapes
    # Output
        sum of the L2 norms of the input weights
        scalar
    """
    
    #################
    ### YOUR CODE ###
    #################
    output = (weight_decay/2)*np.sum(np.power(weights,2))
    return output

class Optimizer(object):
    '''
    This is a basic class. 
    All other optimizers will inherit it
    '''
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        
    def update_params(self):
        pass


class SGD(Optimizer):
    '''
    Stochastic gradient descent optimizer
    https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    '''
        
    def update_params(self):
        weights = self.model.get_params()
        grads = self.model.get_params_gradients()
        for w, dw in zip(weights, grads):
            update = self.lr * dw + self.weight_decay * w
            # it writes the result to the previous variable instead of copying
            np.subtract(w, update, out=w) 
            
def dropout_generate_mask(shape, drop_rate):
    """Generate binary mask 
    # Arguments
        shape: shape of the input array 
            tuple 
        drop_rate: probability of the element 
            to be multiplied by 0
            scalar
    # Output
        binary mask 
    """
    #################
    ### YOUR CODE ###
    #################
    random_sample = np.random.sample(shape)
    m = np.ma.masked_where(random_sample > drop_rate, random_sample)
    mask = m.mask
    return mask

def dropout_forward(x_input, mask, drop_rate, training_phase):
    """Perform the mapping of the input
    # Arguments
        x_input: input of the layer 
            np.array of size `(n_objects, d_in)`
        mask: binary mask
            np.array of size `(n_objects, d_in)`
        drop_rate: probability of the element to be multiplied by 0
            scalar
        training_phase: bool either `True` - training, or `False` - testing
    # Output
        the output of the dropout layer 
        np.array of size `(n_objects, d_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    output = x_input * mask if training_phase else x_input * (1 - drop_rate)
    return output


def dropout_grad_input(x_input, grad_output, mask):
    """Calculate the partial derivative of 
        the loss with respect to the input of the layer
    # Arguments
        x_input: input of a dense layer - np.array of size `(n_objects, n_in)`
        grad_output: partial derivative of the loss functions with 
            respect to the ouput of the dropout layer 
            np.array of size `(n_objects, n_in)`
        mask: binary mask
            np.array of size `(n_objects, n_in)`
    # Output
        the partial derivative of the loss with 
        respect to the input of the layer
        np.array of size `(n_objects, n_in)`
    """
    #################
    ### YOUR CODE ###
    #################
    grad_input = grad_output * mask
    return grad_input


class Dropout(Layer):
    
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.drop_rate = drop_rate
        self.mask = 1.0
        
    def forward(self, x_input):
        if self.training_phase:
            self.mask = dropout_generate_mask(x_input.shape, self.drop_rate)
        self.output = dropout_forward(x_input, self.mask, 
                                      self.drop_rate, self.training_phase)
        return self.output
    
    def backward(self, x_input, grad_output):
        grad_input = dropout_grad_input(x_input, grad_output, self.mask)
        return grad_input
    

def mse_forward(target_pred, target_true):
    """Compute the value of MSE loss
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the value of MSE loss 
        for a given prediction and the ground truth
        scalar
    """
    #################
    ### YOUR CODE ###
    #################
    n = len(target_pred)
    output = (1/(2*n))*np.sum(np.power(target_true - target_pred,2))
    return output

def mse_grad_input(target_pred, target_true):
    """Compute the partial derivative 
        of MSE loss with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects,)`
        target_true: ground truth - np.array of size `(n_objects,)`
    # Output
        the partial derivative 
        of MSE loss with respect to its input
        np.array of size `(n_objects,)`
    """
    #################
    ### YOUR CODE ###
    #################
    n = len(target_pred)
    grad_input = (target_pred - target_true) / n
    return grad_input

class MSE(Loss):
    
    def forward(self, target_pred, target_true):
        self.output = mse_forward(target_pred, target_true)
        return self.output
    
    def backward(self, target_pred, target_true):
        return mse_grad_input(target_pred, target_true)