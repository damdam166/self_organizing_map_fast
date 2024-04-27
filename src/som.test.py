import numpy as np

import som

def test_distance() -> bool:
    print(f'Testing the function called {test_distance.__name__}.')

    X = np.array([ 0 for k in range(9) ])
    Y = np.array([ 1 for k in range(9) ])

    assert som.distance(X, X) == 0
    assert som.distance(X, Y) == 3

    print(f'\tOK.\n')
    return True

def test_learning_rate() -> bool:
    print(f'Testing the function called {test_distance.__name__}.')
    assert som.learning_rate(0, 100, 1) == 1
    assert som.learning_rate(0, 100, 10) == 10 
    assert som.learning_rate(1, 100, 10) <= som.learning_rate(0, 100, 10)
    assert som.learning_rate(2, 100, 10) <= som.learning_rate(1, 100, 10)

    print(f'\tOK.\n')
    return True

def test_neighborhood() -> bool:
    print(f'Testing the function called {test_distance.__name__}.')

    assert som.neighborhood(0, 1) == 1
    assert som.neighborhood(0, 10) == 1
    assert som.neighborhood(1, 10) <= som.neighborhood(0, 10)
    assert som.neighborhood(2, 10) <= som.neighborhood(1, 10)

    print(f'\tOK.\n')
    return True

def test_initialize_neuron() -> bool:
    print(f'Testing the function called {test_distance.__name__}.')

    min = np.array([ -1, 0, -10, 23 ])
    max = np.array([ 10, 6, 0, 42 ])
    X = som.initialize_neuron(min, max)

    assert X.shape[0] == min.shape[0]
    assert min[0] <= X[0] <= max[0]
    assert min[1] <= X[1] <= max[1]
    assert min[2] <= X[2] <= max[2]
    assert min[3] <= X[3] <= max[3]

    print(f'\tOK.\n')
    return True

def test_winner() -> bool:
    print(f'Testing the function called {test_distance.__name__}.')

    X = np.array([ 0 ])
    array_neurons = np.array([ [ 1 ], [ 2 ], [ 10 ] ])
    assert som.winner(X, array_neurons) == 0

    X = np.array([ 11 ])
    assert som.winner(X, array_neurons) == 2

    print(f'\tOK.\n')
    return True

if __name__ == '__main__':
    print(f'\n################')
    print(f'\nTESTING THE SOM.\n')
    print(f'################\n')

    assert test_distance()
    assert test_learning_rate()
    assert test_neighborhood()
    assert test_initialize_neuron()
    assert test_winner()


