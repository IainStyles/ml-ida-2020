with open('Neurips2016Papers.txt','r') as f:
    docs = f.read().split("\n")

print(len(docs))

# Write each of the docs in a file with all of the non-alpha characters stripped
# Necessary because there is a lot of math in here
with open("neurips_abstracts.txt",'w') as doclist:
    for i,d in enumerate(docs):
        with open(f"./neurips_abstracts/neurips-{i}.txt",'w') as f:
            # Strip non-alpha
            s = ''.join([i if i.isalpha() else " " for i in d])
            # Strip short strings letters
            tokens = s.split()
            s = ' '.join([i for i in tokens if len(i)>1 ])
            f.write(s)
        doclist.write(f"./neurips_abstracts/neurips-{i}.txt\n")