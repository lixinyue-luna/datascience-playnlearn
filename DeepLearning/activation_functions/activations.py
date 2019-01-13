import numpy as np

# activation functions and their derivatives
# sigmoid
def sigmoid(x):
    '''
    Compute sigmoid of x
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    
    Return:
    s: sigmoid(x)
    '''
    s = 1/(1 + np.exp(-x))
    return s

def sigmoid_derivative(x):
    '''
    Compute the derivative of sigmoid(x)
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    
    Return:
    ds: derivative of sigmoid fucntion
    '''
    # first calculate the sigmoid function
    s = sigmoid(x)

    # then use formula (2) to compute the derivative
    ds = s * (1-s)
    return ds

# tanh
def tanh(x):
    '''
    Compute tanh of x
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    
    Return:
    t: tanh(x)
    '''
    t = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return t

def tanh_derivative(x):
    '''
    Compute the derivative of tanh(x)
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    
    Return:
    ds: derivative of tanh fucntion
    '''
    # first calculate the sigmoid function
    t = tanh(x)

    # then use formula (2) to compute the derivative
    dt = 1 - t**2
    return dt

# ReLU
def relu(x):
    '''
    Compute relu of x
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    
    Return:
    r: relu(x)
    '''
    r = np.where(x >= 0, x, 0)
    return r

def relu_derivative(x):
    '''
    Compute the derivative of relu(x)
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    
    Return:
    ds: derivative of tanh fucntion
    '''
    # then use formula (2) to compute the derivative
    dr = np.where(x >= 0, 1, 0)
    return dr

# Leaky ReLU
def leaky_relu(x, a=0.01):
    '''
    Compute leaky relu of x
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    a: the 
    
    Return:
    r: relu(x)
    '''
    r = np.where(x >= 0, x, a*x)
    return r

def leaky_relu_derivative(x, a=0.01):
    '''
    Compute the derivative of relu(x)
    
    Arguments:
    x: a scalar or vector/matrix represented as numpy array
    
    Return:
    ds: derivative of tanh fucntion
    '''
    # then use formula (2) to compute the derivative
    dr = np.where(x >= 0, 1, a)
    return dr