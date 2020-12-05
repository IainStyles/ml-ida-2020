import numpy as np
import matplotlib.pyplot as plt
from markov import markovchain
np.set_printoptions(precision=2)

def main():
    # Dangling page example
    p = np.array([.2,.2,.2,.2,.2])
    T = np.array(
        [
            [1./3,0.  ,1./3 ,1./3,0.],
            [0.25,.25 ,0.   ,.25 ,.25],
            [.5  ,0.  ,0.   ,0.  ,.5],
            [0.  ,0.  ,0.   ,0.  ,0.],
            [.25 ,.25 ,0.25 ,.25 ,0.]
        ]
    )
    pt = markovchain(T,p,20)
    print(pt)

    # Dangling page fix
    p = np.array([.2,.2,.2,.2,.2])
    T = np.array(
        [
            [1./3,0.  ,1./3 ,1./3,0.],
            [0.25,.25 ,0.   ,.25 ,.25],
            [.5  ,0.  ,0.   ,0.  ,.5],
            [0.2 ,0.2 ,0.2  ,0.2 ,0.2],
            [.25 ,.25 ,0.25 ,.25 ,0.]
        ]
    )
    pt = markovchain(T,p,20)
    print(pt)

    plt.show()

    return None

if __name__ == '__main__':
    main()