import gensim
from gensim.models import Doc2Vec,KeyedVectors
import pickle
import lib
import numpy as np
# calculate the average distances between the trainings and 
glove = KeyedVectors.load('glove-300')
model = Doc2Vec.load('abs_d2v.model')
abs = pickle.load(open('tests.array','rb'))
suits = pickle.load(open('suitable_abs.array','rb'))
print("full number of papers "+str(len(tests)))
print("full number of abstracts "+str(len(abs)))
def test_dist(model,tests):
    test_distance = []
    for i in range(len(tests)-1):
        dist = (model.wv.wmdistance(tests[i],tests[i+1]))
        test_distance.append(dist)
    return test_distance

print("mean: "+str(np.mean(test_dist(model,abs))))

test_distance=[]
for i in range(len(abs)-1):
    dist = (glove.wmdistance(abs[i],abs[i+1]))
    test_distance.append(dist)
print("Glove mean: "+str(np.mean(test_distance)))