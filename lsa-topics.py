import easyargs
import tf_idf
import doc_vectors
import numpy as np
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
    
    # Now take each of the topics and identify the most important terms

    # Easier to work with the transpose for this because of how the data is organised
    for i,u in enumerate(U[:,0:10].transpose()):
        # Take the absolute value of u and find the ten most significant terms
        print(f"\n\n*** Topic {i} ***")
        print(u)
        absu = list(np.abs(u))
        topic = [term for _,term in sorted(zip(absu,vocabulary), reverse=True)[0:10]]
        print(topic)
        # Find the top-three articles for the topic
        topic_doc_projection = np.abs(np.matmul(u,document_matrix))
        #print(topic_doc_projection)
        best_matches = np.argsort(topic_doc_projection)
        print(f"Three best matches are documents: {best_matches[-1]}, {best_matches[-2]}, {best_matches[-3]}")


    return None

if __name__ == '__main__':
    main()