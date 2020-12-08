import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

def markovchain(T,p,nsteps):
    pt = [p]
    t = np.linspace(1,nsteps,nsteps)
    print(t)
    for i in t:
        p = np.matmul(T.transpose(),p)
        pt.append(p)
    print(f"Last p:{p}")
    pt = np.array(pt)
    t = np.insert(t,0,0)
    plt.figure()
    for i,x in enumerate(p):
        plt.plot(t,pt[:,i], label=f'i={i+1}')
    plt.xlabel('t')
    plt.ylabel(r'$p^{(t)}_i$')
    plt.legend(loc='upper right')  
    w,v = np.linalg.eig(T.T)
    # get eigenvalue closest to 1
    index = np.argmin(abs(w-1.0))
    print(f"Eigenvalue nearest 1: {w[index]}")
    print(f"Corresponding eigenvector: {v[:,index]}")
    print(f"Normalised eigenvector: {v[:,index]/np.sum(v[:,index])}")
    return v[:,index]/np.sum(v[:,index])

def main():
    # Define the initial state
    p = np.array([.1,.1,.3,.2,.3])
    # Define the Transition matrix
    T=np.array(
        [[0,.7,0,0,.3], 
        [0,0,0,1,0], 
        [0,.6,0,.4,0], 
        [.4,.3,.3,0,0], 
        [0,1,0,0,0]]
    )

    pt = markovchain(T,p,20)
    print(pt)

    # A second example
    p = np.array([0.6,0.2,0.2,0])
    T = np.array(
        [[.3,.3,.3,.1],
        [.2,.2,.2,.4],
        [.2,.3,.2,.3],
        [0,0,0,1]]
    )
    pt = markovchain(T,p,20)
    print(pt)

    # Third example
    p = np.array([0.3,0.1,0.6])
    T = np.array(
        [[0,1,0],
        [1,0,0],
        [0,1,0]]
    )
    pt = markovchain(T,p,20)
    print(pt)

    # Fourth Example - Pagerank matrix
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