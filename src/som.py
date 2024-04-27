import numpy as np
import math
import random

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

def neighborhood(dist_x_y : float, s : float = 1) -> float:
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
    return compose(np.arange(min.shape[0]))

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
    are_neighbors = lambda i, j: i == j or i == j - 1 or i == j + 1

    compose = lambda i: ( 
        array_neurons[i] if not are_neighbors(c, j) else ( 
            array_neurons[i] 
                + neighborhood(X_input, array_neurons[i]) 
                    * learning_rate(t, T) 
                    * ( X_input - array_neurons[i] ) 
        ))

    return compose(np.arange(array_neurons.shape[1]))


