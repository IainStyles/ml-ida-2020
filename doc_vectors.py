import easyargs
import tf_idf
import numpy as np
np.set_printoptions(precision=3)

def vectorise(document_tokens, inverse_index, idf):
    # Create a zero-filled vector
    vector = np.zeros((len(inverse_index)))
    for i,term in enumerate(inverse_index):
        vector[i] = document_tokens.count(term) * idf[term]
    return vector

@easyargs
def main(document_list, stopwordfile):
    # Load the document list - one file per line
    with open(document_list,'r') as f:
        print(f"Loading document list from {document_list}")
        doclist = f.read().split()
    # Construct the vocabulary
    vocabulary, document_tokens = tf_idf.build_vocabulary(doclist, stopwordfile)
    print(sorted(vocabulary))
    # Build the inverse index
    inverse_index = tf_idf.build_inverse_index(document_tokens, sorted(vocabulary))
    # Computer the inverse doc frequencies
    idf = tf_idf.calc_idf(inverse_index, len(doclist))

    for doc in document_tokens:
        v = vectorise(doc, inverse_index, idf)
        print(f"{v}")

    return None

if __name__ == '__main__':
    main()