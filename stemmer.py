import easyargs
import nltk.stem as stem
import os.path

def stemmer(text):
    stemmer = stem.PorterStemmer("NLTK_EXTENSIONS")
    stemmed_tokens = [stemmer.stem(token) for token in text.split()]
    return stemmed_tokens

@easyargs
def main(infile):
    if os.path.isfile(infile):
        print(f"Loading text from file {infile}...\n")
        with open(infile,'r') as f:
            text = f.read()
    else:
        print(f"Running stemmer on input word")
        text = infile
    print(f"=============\n{text}\n=============\n\n")

    stemmed_tokens = stemmer(text)

    print(" ".join(stemmed_tokens))

    outfile = infile.split(".")[0] + "-stemmed.txt"

    with open(outfile,'w') as f:
        f.write(" ".join(stemmed_tokens))

    return None
    
if __name__ == '__main__':
    main()