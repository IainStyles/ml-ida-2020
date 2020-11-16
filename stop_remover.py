import easyargs

def stop_remover(text, stopwordsfile):
    # Load the list of stopwords
    with open(stopwordsfile,'r') as f:
        stopwords = f.read().lower().split()

    # Define the punctuation characters
    punctuation = [i for i in ',./\\;:\'@#~[{]}=+-_)(*&^%$£"!?`)]']

    # Deal with plurals as a special case, replacing with white space
    text = text.replace("'s"," ")

    # Replace all punctuation with white space and make lower case
    text = "".join([" " if t in punctuation else t for t in text]).lower()

    # Split the text on arbitrary quantity of whitespace
    tokens = text.split()

    # Remove the stop word
    tokens = [token for token in tokens if token not in stopwords]
    text = " ".join(tokens)
    return text


@easyargs
def main(textfile, stopwordsfile):
    print(f"Loading text from file {textfile}...\n")
    with open(textfile,'r') as f:
        text = f.read()
    print(f"=============\n{text}\n=============\n\n")

    text = stop_remover(text, stopwordsfile)
    print(text)

    outfile = textfile.split(".")[0] + "-stop.txt"
    with open(outfile,'w') as f:
        f.write(text)

    return None

if __name__ == '__main__':
    main()
    