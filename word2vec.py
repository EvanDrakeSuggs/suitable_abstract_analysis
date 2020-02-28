import os
import re
import pickle
#dir = "artio_corpus/"
entry = os.listdir();
dir_suitable = [x for x in entry  if "suitable.txt" in x]
print(dir_suitable);

def entry_split(entry):
    exp = re.compile("\n[A-Z][A-Z]")
    labels = [entry[:3]]+exp.findall(entry)
    labels = [label[1:] for label in labels]
    entry_values = exp.split(entry)
    entry_dictionary = dict(zip(labels,entry_values))
    return entry_dictionary


for file in dir_suitable:
    print(file)
    parse = open("artio_corpus/suitable_artio.txt",'r')
    parse = re.split(r"\nER\n",parse.read())
    entries = []
    abstracts = []
    for entry in parse:
        #abstracts.append((entry_split(entry)['AB']))
        entries.append(entry)
        
        #print(entry_split(entry))
        #if(input("Want to break ('y'): ") == "y"):
         #  break
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

print("length: "+str(len(entries)))
# tagging data
#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),tags=[str(i)]) for i, _d in enumerate(entries)]
words = ([entry.split() for entry in entries])

max_epochs = 100
vec_size = 20
alpha = 0.025
model = Word2Vec(data,size=100,window=5,min_count=1,workers=4)
model.build_vocab(words)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    #decrease the learning rate
    model.alpha -= 0.0002
    #
    model.min_alpha = model.alpha

model.save("word2vec.model")
print("Model Saved")


# some sort of distance/similarty, either cosine (gensimmatutils.cossim) or hellinger distance
from scipy import spatial
diff = spatial.distance.cosine
