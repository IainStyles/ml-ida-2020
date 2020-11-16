import easyargs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

@easyargs
def main(infile):
    print(f"Loading word frequencies from file {infile}...\n")
    word_freq = pd.read_csv(infile, header=None)

    words = word_freq.iloc[:,0].to_numpy()
    counts = word_freq.iloc[:,1].to_numpy()
    rank = np.linspace(1, words.size+1, words.size)

    # Fit Zipf's law to the data
    logcounts = np.log(counts)
    logrank = np.log(rank)

    # This uses a quick analytic solution of the least squares problem for y=mx+c for speed

    Sx = np.mean(logrank)
    Sy = np.mean(logcounts)
    Sxx = np.mean(logrank*logrank)
    Sxy = np.mean(logrank*logcounts)

    m = (Sxy - Sx*Sy) / (Sxx - Sx*Sx)
    c = Sy - m*Sx

    alpha = -m
    intercept = np.log(c)
       
    plt.figure()
    plt.title(f"Word frequencies in {infile}\n" + r"$\alpha = $" + f"{alpha:.3f}; " + r"$C = $" + f"{intercept:.3f}")
    plt.ylabel("$\log(F(r))$")
    plt.xlabel("$\log(r)$")
    plt.scatter(logrank, logcounts, color='blue', marker=".")
    plt.plot(logrank, m*logrank + c, color='red')
    plt.savefig(f"{infile.split('.')[0]}-zipf.png")
    plt.show()

    return None

if __name__ == '__main__':
    main()