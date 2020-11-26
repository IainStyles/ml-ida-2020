from stop_remover import stop_remover
from stemmer import stemmer
from word_frequency import word_frequency
import numpy as np
import easyargs

def sim(query_wf, doc_number, inverse_index, idf):
    query_weight = dict()
    doc_weight = dict()
    for term in inverse_index:
        query_weight[term] = query_wf.get(term,0) * idf[term]
        # get the weight of the term in the document, WF=0 if not present
        doc_weight[term] = inverse_index[term].get(doc_number,0) * idf[term]
    # vector of query weights
    qw = np.array([query_weight[term] for term in inverse_index])
    # vector of document weights
    dw = np.array([doc_weight[term] for term in inverse_index])
    
    if np.linalg.norm(dw) == 0:
        simqd = 0
    else:
        simqd = np.dot(qw,dw)/(np.linalg.norm(qw) * np.linalg.norm(dw))
    return simqd

def calc_idf(inverse_index, num_docs):
    idf = dict()
    for term in inverse_index:
        idf[term] = np.log(num_docs/len(inverse_index[term]))
    return idf

def build_inverse_index(document_tokens, vocabulary):
    # Construct a dictionary with the vocab as the key and an empty list as the value
    inverse_index = {term:dict() for term in vocabulary}
    for document_number, tokens in enumerate(document_tokens):
        print(f"Adding document {document_number} to the inverse index")
        for term in vocabulary:
            termcount = tokens.count(term)
            if termcount > 0:
                inverse_index[term][document_number] = termcount
    return inverse_index

def build_vocabulary(doclist, stopwordfile):
    # Create an empty set for the vocabulary
    print("Building the vocabulary")
    vocabulary = set()
    document_tokens = []
    for doc in doclist:
        print(f"Loading document {doc}")
        with open(doc,'r') as f:
            text = f.read()
            # Remove stop words
            text = stop_remover(text, stopwordfile)
            # stem
            tokens = stemmer(text)
            document_tokens.append(tokens)
            unique_tokens = set(tokens)
            print(f"{len(unique_tokens)} unique tokens")
            # Add the tokens to the vocab if they are not already there.
            vocabulary = vocabulary.union(unique_tokens)
            print(f"There are now {len(vocabulary)} words in the vocabulary\n")
    return vocabulary, document_tokens


@easyargs
def main(query_file, document_list, stopwordfile):
    # Load the document list - one file per line
    with open(document_list,'r') as f:
        print(f"Loading document list from {document_list}")
        doclist = f.read().split()
    # Construct the vocabulary
    vocabulary, document_tokens = build_vocabulary(doclist, stopwordfile)
    # Build the inverse index
    inverse_index = build_inverse_index(document_tokens, vocabulary)
    # Computer the inverse doc frequencies
    idf = calc_idf(inverse_index, len(doclist))
    # Load the query
    with open(query_file,'r') as f:
        query = f.read()
        # stop and stem the query
        query_text = stop_remover(query, stopwordfile)
        query_tokens = stemmer(query_text)
        # compute the word frequencies in the query
        query_wf = dict()
        for term in set(query_tokens):
            query_wf[term] = query_tokens.count(term)

        print(f"\nQuery word frequencies")
        print(query_wf)

    # Now we compute the similarity with each document
    print("\nComputing similarity of query with documents")
    for doc_number in range(len(doclist)):
        similarity = sim(query_wf, doc_number, inverse_index, idf)
        print(f"sim(q,d{doc_number}) = {similarity:.2f}")

    return None

if __name__ == '__main__':
    main()
