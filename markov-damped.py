import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

def dampedmarkovchain(T,p,nsteps,gamma):
    pt = [p]
    t = np.linspace(1,nsteps,nsteps)
    print(t)
    for i in t:
        p = (1-gamma)*np.matmul(T.transpose(),p) + (gamma/p.size)*np.ones(p.shape)
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
    p = np.array([0.3,0.1,0.6])
    T = np.array(
        [[0,1,0],
        [1,0,0],
        [0,1,0]]
    )
    pt = dampedmarkovchain(T,p,nsteps=30,gamma=0)
    print(pt)

    
    ptd = dampedmarkovchain(T,p,nsteps=100,gamma=0.1)
    print(ptd)


    plt.show()

    return None

if __name__ == '__main__':
    main()