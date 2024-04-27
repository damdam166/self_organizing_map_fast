import numpy as np
import math
import random
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# DISTANCE
# -------------------------------------------------------------------------

def distance(X : np.ndarray, Y : np.ndarray) -> float:
    """ Do dist(X, Y) for their dimension, using the euclidean distance.
    :param X,Y: 2 vectors from R^n, same dimension.
    :return: dist(X, Y).
    """
    # return math.sqrt(sum([ ( X[i] - Y[i] ) ** 2 
    # for i in range(X.shape[0]) ]))

    compose = lambda index: ( X[index] - Y[index] ) ** 2
    return np.sqrt(
        np.sum(
            compose(np.arange(X.shape[0]))
        )
    )

# -------------------------------------------------------------------------
# LEARNING RATE
# -------------------------------------------------------------------------

def learning_rate(t : int, T : int = 100, a : float = 1) -> float:
    """ function alpha : t |--> alpha(t) that decrease, defined on N. 
    :param T: the number of iterations.
    :param a: alpha(0) = a.
    """
    return a * math.exp( - t / T )

# -------------------------------------------------------------------------
# NEIGHBORHOOD FUNCTION
# -------------------------------------------------------------------------

def neighborhood(dist_x_y : float, s : float = 5) -> float:
    """ f(dist_x_y) == 1 <==> x and y are neighbors.
    :param dist_x_y: dist(X, Y) where X is an input vector, Y a neuron.
    """
    return math.exp( - dist_x_y * dist_x_y / ( 2 * s * s ) )

# -------------------------------------------------------------------------
# CREATE A NEURON
# -------------------------------------------------------------------------

def initialize_neuron(min : np.ndarray, max : np.ndarray) -> np.ndarray:
    """ Returns V such as V_i in [min_i, max_i].
    :param min, max: vectors of same dimesions.
    :return: a random V.
    """
    # return np.array([ 
    # random.random() * ( max[i] - min[i] ) + min[i] 
    # for i in range(min.shape[0]) 
    # ])

    compose = lambda i: random.random() * ( max[i] - min[i] ) + min[i]
    return np.array([ compose(i) for i in range(min.shape[0]) ])

# -------------------------------------------------------------------------
# WINNER
# -------------------------------------------------------------------------

def winner(X_input : np.ndarray, array_neurons : np.ndarray) -> int:
    """ Find the best machine unit.
    :param X_input: the input vector.
    :param array_neurons: an array of neurons, same dimen with X_input.
    :return: the index of the winner from the array_neurons.
    """
    compose = lambda neuron: distance(X_input, neuron)
    return np.argmin(
        np.array([ compose(neuron) for neuron in array_neurons ])
    )

# -------------------------------------------------------------------------
# UPDATE
# -------------------------------------------------------------------------

def update(t : int, c : int, X_input : np.ndarray,
           array_neurons : np.ndarray, 
           T : int = 100) -> np.ndarray:
    """ Update the neurons.
    :param t: the current iteration.
    :param T: the total number of iterations.
    :param c: the index in array_neurons of the winner.
    :param X_input: the input vector.
    :param array_neurons: an array of neurons, same dimen with X_input.
    """

    # Linear topology
    are_neighbors = lambda i: i == c or i == c - 1 or i == c + 1

    compose = lambda i: ( 
        array_neurons[i] if not are_neighbors(i) else ( 
            array_neurons[i] 
                + neighborhood(distance(X_input, array_neurons[i])) 
                    * learning_rate(t, T) 
                    * ( X_input - array_neurons[i] ) 
        )
    )

    return np.array([ compose(i) for i in range(array_neurons.shape[0]) ])

# -------------------------------------------------------------------------
# PLOT
# -------------------------------------------------------------------------

def display(array_inputs : np.ndarray, array_neurons : np.ndarray, 
            c : int = -1, X : np.ndarray = None):
    """ To plot the map with Matplotlib.
    :param array_inputs: the array of input vectors.
    :param array_neurons: the neurons.
    They are vectors of same dimension with array_inputs.
    :param c: the index of winner in array_neurons.
    :param X: the chosen input vector if necessary.
    -1 means there is no need to display the winner.
    """
    fig, ax = plt.subplots()
    
    # Input vectors
    plt.plot(
        array_inputs[:, 0],
        array_inputs[:, 1],
        marker='o',
        color='blue',
    )

    # Chosen input vector
    if np.all(X) != None:
        plt.plot(
            X[0],
            X[1],
            marker='o',
            markersize=10,
            color='green',
        )

    # Map
    plt.plot(
        array_neurons[:, 0],
        array_neurons[:, 1],
        marker='o',
        markersize=4,
        color='red',
    )

    # Winner
    if c != - 1:
        plt.plot(
            array_neurons[c][0],
            array_neurons[c][1],
            marker='o',
            markersize=10,
            color='black',
        )

    plt.axis(False)
    plt.show()

# -------------------------------------------------------------------------

if __name__ == '__main__':
    # Initialize the input vector
    array_inputs : np.ndarray = np.array([ [10, 10], [5, 10] ])

    # Initialize the neurons
    min = np.array([ 0, 0 ])
    max = np.array([ 5, 5 ])
    number_neurons : int = 5
    compose = lambda i: initialize_neuron(min, max)
    array_neurons = np.array([ compose(0) for i in range(number_neurons) ])

    input = lambda i: array_inputs[random.randrange(array_inputs.shape[0])]

    display(array_inputs, array_neurons)

    # Loop
    T : int = 100 # Number of iterations
    for t in range (T):
        # Choose the input vector
        X = input(0)
    
        # Display the chosen input vector
        display(array_inputs, array_neurons, -1, X)

        # The winner
        c : int = winner(X, array_neurons)

        # Display the winner
        display(array_inputs, array_neurons, c, X)

        # Update
        array_neurons = update(t, c, X, array_neurons, T)

        # Display the update
        display(array_inputs, array_neurons)

# -------------------------------------------------------------------------


