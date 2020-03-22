# made with https://towardsdatascience.com/implementing-multi-class-text-classification-with-doc2vec-df7c3812824d
import os
import re
import pickle
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import lib
from sklearn.linear_model import LogisticRegression
from sklearn import utils
# 
dir_suitable = ['artio_corpus/suitable_artio.txt', 'lago_corpus/suitable_lago.txt']
suitables = lib.to_array(dir_suitable)
suitables = [lib.entry_split(entry) for entry in suitables]
suitable_abs = [entry['AB'] if 'AB' in entry.keys() else "" for entry in suitables ]
tagged_suitables = [TaggedDocument(words=word_tokenize(_d.lower()),tags=[str(i),'suitable']) for i, _d in enumerate(suitable_abs)]

files = os.listdir("artio_corpus/Artiodactyl Unsuitable");
dir_unsuitable = ["artio_corpus/Artiodactyl Unsuitable/"+file for file in files]
files = os.listdir('lago_corpus/Lagomorph Unsuitable')
files = ['lago_corpus/Lagomorph Unsuitable/'+file for file in files]
dir_unsuitable+=files
unsuitables = lib.to_array(dir_unsuitable)
unsuitables = [lib.entry_split(entry) for entry in unsuitables]
unsuitable_abs = [entry['AB'] if 'AB' in entry.keys() else "" for entry in unsuitables]
tagged_unsuitables = [TaggedDocument(words=word_tokenize(_d.lower()),tags=[str(i),'unsuitable']) for i,_d in enumerate(unsuitable_abs)]


train_documents = tagged_suitables[:-50]+tagged_unsuitables[:-50];
test_documents =  tagged_suitables[-50:]+tagged_unsuitables[-50:];

max_epochs = 30
vec_size = 300
alpha = 0.025
model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1,
                workers=4)
model.build_vocab(train_documents) #+test_documents)
print(len(model.wv.vocab))
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(train_documents,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    #decrease the learning rate
    model.alpha -= 0.0002
    #
    model.min_alpha = model.alpha

print("training done")
#model.save('models/cls.model')
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors

y_train, X_train = vector_for_learning(model, train_documents)
y_test, X_test = vector_for_learning(model, test_documents)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Testing accuracy for movie plots%s' % accuracy_score(y_test, y_pred))
print('Testing F1 score for movie plots: {}'.format(f1_score(y_test, y_pred, average='weighted')))