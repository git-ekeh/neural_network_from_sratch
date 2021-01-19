
import numpy as np

def sigmoid(x):
    '''
    The sigmoid function is an activation function that squishes the numbers between 1 and 0.
    This is neccessary for a binary prediction algorithm.
    '''
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 -x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self): #This is how to get the output of the neural network
        '''
        This Neural Network will take an input layer as x
        We use Numpy Arrays with random values
        '''
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        '''
        loss function with respect to weights1 one and weights2
        Backpropagation basically readjusts the weights and biases to their optimal levels
        If we have the derivative, then we can simply update the weights and biases increasing or decreasing it
        This process is known as gradient descent
        Understand the chain rule, because it is important to gradient descent
        '''
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) *
                     sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output)
                     * sigmoid_derivative(self.output), self.weights2.T)*
                       sigmoid_derivative(self.layer1)))
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    X = np.array([[0,0,1],
                 [0,1,1],
                 [1,0,1],
                 [1,1,1]])

    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
