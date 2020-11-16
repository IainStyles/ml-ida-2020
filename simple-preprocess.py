import easyargs

@easyargs
def main(infile):
    print(f"Loading text from file {infile}...\n")
    with open(infile,'r') as f:
        text = f.read()
    print(f"=============\n{text}\n=============\n\n")

    # Define the punctuation characters
    punctuation = [i for i in ',./\\;:\'@#~[{]}=+-_)(*&^%$£"!`)]']

    # Deal with plurals as a special case, replacing with white space
    text = text.replace("'s"," ")

    # Replace all punctuation with white space and make lower case
    text = "".join([" " if t in punctuation else t for t in text]).lower()
    print(f"====== Cleaned Text ======= \n{text}\n=============\n")

    outfile = infile.split(".")[0] + "-simple.txt"
    print(f"Writing results to {outfile}")
    with open(outfile,'w') as f:
        f.write(text)

    return None

if __name__ == '__main__':
    main()
