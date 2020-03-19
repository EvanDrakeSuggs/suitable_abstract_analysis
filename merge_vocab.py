import lib
from gensim.models import KeyedVectors,Doc2Vec

glove = KeyedVectors.load('glove-300')
hayes = Doc2Vec.load('abs.model')

for word in glove.wv.vocab:
    if word not in hayes.wv.vocab:
        hayes.wv.add(word) = glove[word]
print(len(hayes.wv.vocab))