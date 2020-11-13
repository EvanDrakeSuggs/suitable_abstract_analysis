"""
Preprocessing for classifier files
"""
from __future__ import unicode_literals, print_function
import os
import random
import pickle
import thinc.extra.datasets
import spacy
from spacy.util import minibatch, compounding
import lib

def remove_stopwords(abs, nlp):
    """
    removes stopword identified by spacy
    """
    doc = nlp(abs)
    out_filtered=[]
    for word in doc:
        if word.is_stop==False:
            out_filtered.append(word.text)
    return " ".join(out_filtered)

def lemmization(abs, nlp):
    """
    returns doc and turns into 
    """
    doc = nlp(abs)
    out_lemma=[]
    for word in doc:
        out_lemma.append(word.lemma_)
    return " ".join(out_lemma)

# import suitable files from directory
dir_suitable = ['artio_corpus/suitable_artio.txt', 'lago_corpus/suitable_lago.txt']
# use project's file RIS manipulator library to form an array of papers then split them correctly
suitables = lib.to_array(dir_suitable)
suitables = [lib.entry_split(entry) for entry in suitables]
# only keep abstracts for training here then create the categories (cats) tag for each
suitable_abs = [entry['AB'] if 'AB' in entry.keys() else "" for entry in suitables]
suitable_cats = [{"SUITABLE": True, "UNSUITABLE":False} for x in suitable_abs]

# process for suitables is similar but we have two directories, artiocactyl and lagomorphs
artiodactyl_files = os.listdir("artio_corpus/Artiodactyl Unsuitable")
dir_unsuitable = ["artio_corpus/Artiodactyl Unsuitable/"+file for file in artiodactyl_files]
lagomorph_files = ['lago_corpus/Lagomorph Unsuitable/'+file for file \
                   in os.listdir('lago_corpus/Lagomorph Unsuitable')]
dir_unsuitable+=lagomorph_files
unsuitables = [lib.entry_split(entry) for entry in lib.to_array(dir_unsuitable)]
unsuitable_abs = [entry['AB'] if 'AB' in entry.keys() else "" for entry in unsuitables]
unsuitable_cats = [{"SUITABLE":False, "UNSUITABLE":True} for x in unsuitable_abs]

#slicing the data into train and train
train_documents = suitable_abs[:-100]+unsuitable_abs[:-100]
train_cats = suitable_cats[:-100]+unsuitable_cats[:-100]
test_documents =  suitable_abs[-100:]+unsuitable_abs[-100:]
test_cats = suitable_cats[-100:]+unsuitable_cats[-100:]

# ? lemmaization and stop words removal ?
preproc = spacy.load("en_core_web_md")
#train_docs = [remove_stopwords(doc, preproc) for doc in train_documents]
train_docs = [lemmization(doc, preproc) for doc in train_documents]

#train_data = list(zip(train_data, [{"cats": cats} for cats in train_cats]))
with open("pickles/training", "wb") as file:
    pickle.dump([train_docs, train_cats], file)

y_test = [1 for i in range(100)]+[0 for i in range(100)]
with open("pickles/classifier_data", "wb") as file:
    pickle.dump([test_documents,test_cats, y_test], file)

#pickle.dump(preproc, open("pickles/nlp.pickle", "wb"))
#pickle.dump(train_data, open("pickles/data.pickle","wb"))
