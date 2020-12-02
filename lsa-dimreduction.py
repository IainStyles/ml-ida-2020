import easyargs
import tf_idf
import doc_vectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
np.set_printoptions(precision=3, suppress=True)

@easyargs
def main(document_list, stopwordfile):
    # Load the document list - one file per line
    with open(document_list,'r') as f:
        print(f"Loading document list from {document_list}")
        doclist = f.read().split()
    # Construct the vocabulary
    vocabulary, document_tokens = tf_idf.build_vocabulary(doclist, stopwordfile)
    vocabulary = sorted(vocabulary)
    # Build the inverse index
    inverse_index = tf_idf.build_inverse_index(document_tokens, vocabulary)
    # Computer the inverse doc frequencies
    idf = tf_idf.calc_idf(inverse_index, len(doclist))

    # Build the document matrix
    document_matrix = np.zeros((len(inverse_index), len(document_tokens)))
    for i,doc in enumerate(document_tokens):
        document_matrix[:,i] = doc_vectors.vectorise(doc, inverse_index, idf)
    print(f"Document matrix size is {document_matrix.shape}")

    # Perform SVD. Note that np gives S as a vector of the diagonal and return V.T rather than V
    print("\n** Full SVD **")
    U, s, Vt = np.linalg.svd(document_matrix, full_matrices=False)
    print(f"U:{U.shape}; s:{s.shape}; Vt:{Vt.shape}")
    reconstructionError = np.linalg.norm(document_matrix - np.matmul(U, np.matmul(np.diag(s), Vt)))
    print(f"Full SVD reconstruction error: {reconstructionError}")
    # Plot the singular values
    plt.figure()
    plt.plot(np.linspace(1,s.size,s.size).astype(int),s,'bx')
    plt.title(f"Singular Values of Full rank SVD\nData from {document_list}")
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\sigma(i)$")
    plt.savefig("singularvalues.png")

    # Now we try the truncated SVD; we do this manually here
    print("\n ** Truncated SVD **")
    ncomponents = np.linspace(1,s.size,s.size).astype(int)
    truncatedSVDError = []
    print(ncomponents)
    for i in ncomponents:
        Up = U[:,0:i]
        sp = s[0:i]
        Vtp = Vt[0:i:]
        print(f"U:{Up.shape}; s:{sp.shape}; Vt:{Vtp.shape}")
        truncatedSVDError.append(np.linalg.norm(document_matrix - np.matmul(Up, np.matmul(np.diag(sp), Vtp))))
        print(f"Truncated SVD (K={i}) reconstruction error: {truncatedSVDError[-1]}")
    print(truncatedSVDError)
    plt.figure()
    plt.plot(ncomponents,truncatedSVDError,'bx')
    plt.title(f"Reconstruction Error from Truncated SVD\nData from {document_list}")
    plt.xlabel(r"$K$")
    plt.ylabel(r"$\mathrm{norm}(\mathbf{D}_K-\mathbf{U}_K\mathbf{\Sigma}_K\mathbf{V}_K^\mathrm{T}$)")
    plt.savefig("reconstructionerror.png")
    plt.show()

    return None

if __name__ == '__main__':
    main()