import os
import re
import pickle
#dir = "artio_corpus/"
entry = os.listdir()
dir_suitable = [x for x in entry  if "unsuit" in x]
files = os.listdir("artio_corpus/Artiodactyl Unsuitable");
dir_unsuitable = ["artio_corpus/Artiodactyl Unsuitable/"+file for file in files]
print(dir_suitable)
print(dir_unsuitable)

def entry_split(entry):
    # every portion of the web of science data begins with two capital letters
    exp = re.compile("\n[A-Z][A-Z]")
    labels = [entry[:3]]+exp.findall(entry)
    labels = [label[1:] for label in labels]
    entry_values = exp.split(entry)
    entry_dictionary = dict(zip(labels,entry_values))
    return entry_dictionary

# turn to function
def to_array(directory_list):
    entries = []
    for file in directory_list:
        print(file)
        parse = open(file,'r')
        parse = re.split(r"\nER\n",parse.read())
        for entry in parse:
            #entries.append(entry_split(entry))
            entries.append(entry)
            
            #print(entry_split(entry))
            #if(input("Want to break ('y'): ") == "y"):
            #  break
    entries = [entry for entry in entries if entry != '\nEF']
    return entries

entries = to_array(dir_unsuitable)
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
print("length: "+str(len(entries)))
entries = entries[10:]
tests = entries[:10]

pickle.dump(tests,open("tests.array",'wb'))

# tagging data
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),tags=[str(i)]) for i, _d in enumerate(entries)]
tagged_test = [TaggedDocument(words=word_tokenize(_d.lower()),tags=[str(i)]) for i, _d in enumerate(tests)]
max_epochs = 100
vec_size = 20
alpha = 0.025
model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    #decrease the learning rate
    model.alpha -= 0.0002
    #
    model.min_alpha = model.alpha

model.save("unsuit_d2v.model")
print("Model Saved")
#print(model.infer_vector(tests[0].split()))

# some sort of distance/similarty, either cosine (gensimmatutils.cossim) or hellinger distance



""" 
class Corpus():
    def __init__(self,file_object):
        self.file = file_object;
        self.entries = [];

    def __iter__(self):
        self.parse = re.split(r"ER",self.file.read());
        for entry in self.parse:
            yield entry
    
    def __next__(self):"""
        

