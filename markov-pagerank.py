import numpy as np
import matplotlib.pyplot as plt
from markov import markovchain
np.set_printoptions(precision=2)

def main():
    # Pagerank matrix
    p = np.array([0.2,0.2,0.2,0.2,0.2])
    T = np.array(
        [
            [0,.5,0,0,.5],
            [0,0,0,1,0],
            [0,.5,0,.5,0],
            [1./3,1./3,1./3,0,0],
            [0,1,0,0,0]
        ]
    )
    
    pt = markovchain(T,p,20)
    print(pt)

    plt.show()

    return None

if __name__ == '__main__':
    main()