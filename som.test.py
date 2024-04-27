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


if __name__ == '__main__':
    print(f'\n################')
    print(f'\nTESTING THE SOM.\n')
    print(f'################\n')

    assert test_distance()
    assert test_learning_rate()


