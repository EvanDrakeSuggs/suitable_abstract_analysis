"""
Spacy classifier
"""
from __future__ import unicode_literals, print_function
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import utils
from pathlib import Path
import thinc.extra.datasets
import spacy
from spacy.util import minibatch, compounding
import lib

model=None
output_dir=None
n_iter=20
n_texts=500#1000
init_tok2vec = None

if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}
nlp = spacy.blank("en")
textcat = nlp.create_pipe("textcat", 
                          # architectures can be ensemble, simple_cnn or bow
                          config={"exclusive_classes": True, "architecture": "emsemble"})
nlp.add_pipe(textcat, last=True)
textcat.add_label("SUITABLE")
textcat.add_label("UNSUITABLE")
dir_suitable = ['artio_corpus/suitable_artio.txt', 'lago_corpus/suitable_lago.txt']
suitables = lib.to_array(dir_suitable)
suitables = [lib.entry_split(entry) for entry in suitables]
suitable_abs = [entry['AB'] if 'AB' in entry.keys() else "" for entry in suitables ]
suitable_cats = [{"SUITABLE": True, "UNSUITABLE":False} for x in suitable_abs]

files = os.listdir("artio_corpus/Artiodactyl Unsuitable");
dir_unsuitable = ["artio_corpus/Artiodactyl Unsuitable/"+file for file in files]
files = os.listdir('lago_corpus/Lagomorph Unsuitable')
files = ['lago_corpus/Lagomorph Unsuitable/'+file for file in files]
dir_unsuitable+=files
unsuitables = [lib.entry_split(entry) for entry in lib.to_array(dir_unsuitable)]
unsuitable_abs = [entry['AB'] if 'AB' in entry.keys() else "" for entry in unsuitables]
unsuitable_cats = [{"SUITABLE":False, "UNSUITABLE":True} for x in unsuitable_abs]

train_documents = suitable_abs[:-100]+unsuitable_abs[:-100]
train_cats = suitable_cats[:-100]+unsuitable_cats[:-100]
train_data = list(zip(train_documents, [{"cats": cats} for cats in train_cats]))
test_documents =  suitable_abs[-100:]+unsuitable_abs[-100:]
test_cats = suitable_cats[-100:]+unsuitable_cats[-100:]

pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    if init_tok2vec is not None:
        with init_tok2vec.open("rb") as file_:
            textcat.model.tok2vec.from_bytes(file.read())
    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
    batch_sizes = compounding(4.0, 32.0, 1.001)
    for i in range(n_iter):
        losses = {}
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        with textcat.model.use_params(optimizer.averages):
            # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, test_documents, test_cats)
            print(
                    "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                            losses["textcat"],
                            scores["textcat_p"],
                            scores["textcat_r"],
                            scores["textcat_f"],
                    )
            )
if output_dir is not None:
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    print(test_text, doc2.cats)

nlp.to_disk("textcat")
test_text="this species is wild and crazy"
doc = nlp(test_text)
print(test_text, doc.cats) 
print("testing")
#nlp = nlp.from_disk("textcat")
for test_text, text_cat in zip(test_documents, test_cats):
    doc = nlp(test_text)
    print(test_text)
    print(text_cat, doc.cats)
'''
#model = Doc2Vec.load('models/cls.model')
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[1], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors

y_train, X_train = vector_for_learning(model, train_documents)
y_test, X_test = vector_for_learning(model, test_documents)

logreg = LogisticRegression(n_jobs=1, C=1e5,verbose=1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
 
print('Testing accuracy for movie plots %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score for movie plots: {}'.format(f1_score(y_test, y_pred, average='weighted')))
'''
