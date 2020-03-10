import os
import re
import pickle
import gensim
import gensim.downloader as api
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize import word_tokenize

#import lib

#model = Doc2Vec.load("d2v.model")
#rint("begin load")
#model = api.load("glove-wiki-gigaword-300")
#model.save('glove-wiki.model')

#print(model.wv.vocab)
important_terms = ['social','group','herd','territorial','solitary','bachelor','plural','singular','aggregation','gregarious']
secondary_imp = ['juveniles','infants','paternity','site','sites']
#web_category = ['behavioral science', 'zoology','environmental science','ecology']
web_category = ['zoology','ecology']
web_topic = ['social', 'herd', 'group','groups','habitats','habitat','environment','environments']
#model = api.load('glove-wiki-gigaword-300')
model = KeyedVectors.load('glove-300')
words = important_terms+secondary_imp+web_category+web_topic
word_similarity = {}
for word in words:
    similars = model.most_similar(word)
    word_similarity[word]=similars

import lib

suitables = ["artio_corpus/suitable_artio.txt","lago_corpus/suitable_lago.txt"]
suits = lib.to_array(suitables)
print(len(suits))

suitability = []

