import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    
    #Computing the first round of forward propagation by calculating z1 
    #z1 will be equal to the data and our parameters W1
    z1 = np.dot(data,W1) + b1

    #USing the sigmoid function for the first layer
    a1 = sigmoid(z1)

    #Taking the answer from the sigmoid function and dot product with the new parameters W2
    z2 = np.dot(a1,W2) + b2

    #FInal output layer gets taken with the softmax function 
    a2 = softmax(z2)


    #Finally, computing the cost function as:
    cost = - np.sum(labels * np.log(a2))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation

    #The first derivative 
    dz2 = a2 - labels

    #THe derivative dz1 will be equal to d2 and the dot product with W2 cells as per chain rule 
    dz1 = np.dot(dz2,W2.T)
    dz0 = np.multiply(dz1, sigmoid_grad(a1))




    #AS per the final derivatives :
    gradW1 = np.dot(data.T , dz0)
    gradb1 = np.sum(dz0, axis=0)
    gradW2 = np.dot(a1.T , dz2)
    gradb2 = np.sum(dz2, axis=0)



    ### END YOUR CODE
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
