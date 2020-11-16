import easyargs
import matplotlib.pyplot as plt
import numpy as np

def word_frequency(text):
    # Split the text on arbitrary quantity of whitespace
    tokens = text.split()
    # Get the unique list of words
    words = list(set(tokens))
    # Generate a list of tuples of (word, count) sorted from high to low
    wordcounts = sorted([(w,tokens.count(w)) for w in words], key=lambda x:x[1], reverse=True)
    return wordcounts

@easyargs
def main(infile):
    print(f"Loading text from file {infile}...\n")
    with open(infile,'r') as f:
        text = f.read()
    print(f"=============\n{text}\n=============\n\n")

    wordcounts = word_frequency(text)

    # Write the results to a csv file
    outfile = infile.split(".")[0] + "-wf.csv"

    print(f"Writing results to {outfile}")
    with open(outfile,'w') as f:
        for w in wordcounts:
            f.write(f"{w[0]},{w[1]}\n")

    # Extract words and counts into two lists
    words = [x[0] for x in wordcounts]
    counts = [x[1] for x in wordcounts]
    # Construct the rank starting at 1
    rank = [i+1 for i,x in enumerate(wordcounts)]

    plt.figure()
    plt.title(f"Word frequencies in {infile}")
    plt.ylabel("$F(r)$")
    plt.xlabel("$r$")
    plt.scatter(rank, counts, color='blue', marker=".")
    plt.savefig(f"{outfile.split('.')[0]}.png")
    plt.show()

    return None

if __name__ == '__main__':
    main()