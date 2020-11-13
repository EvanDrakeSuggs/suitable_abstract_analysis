"""
Spacy classifier
"""
from __future__ import unicode_literals, print_function
import os
import random
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import utils
import thinc.extra.datasets
import spacy
from spacy.util import minibatch, compounding
import lib

spacy.prefer_gpu()
model=None
output_dir=None
n_iter=30
n_texts=500#1000
init_tok2vec = None

if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

def remove_stopwords(doc):
    """
    removes stopword identified by spacy
    """
    out_filtered=[]
    for word in doc:
        if word.is_stop==False:
            out_filtered.append(word.text)
    return " ".join(out_filtered)

def evaluate(tokenizer, textcat, texts, cats, tp=0.0, fp=1e-8, fn=1e-8, tn=0.0):
    """
    Evaluates to show scores
    tp True positives
    fp False positives
    fn False negatives
    tn True negatives
    """
    docs = (tokenizer(text) for text in texts)
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
#preproc = spacy.load("en_core_web_md")
textcat = nlp.create_pipe("textcat",
                          # architectures can be ensemble, simple_cnn or bow
                          config={"exclusive_classes": True, "architecture": "ensemble"})
nlp.add_pipe(textcat, last=True)
textcat.add_label("SUITABLE")
textcat.add_label("UNSUITABLE")

#unpickle preprocessed data
with open("pickles/training", "rb") as file:
    train_docs, train_cats = pickle.load(file)

#train_docs = [remove_stopwords(doc) for doc in train_docs]
train_data = list(zip(train_docs, [{"cats":cats} for cats in train_cats]))

with open("pickles/classifier_data", "rb") as file:
    test_documents,test_cats, y_test = pickle.load(file)

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
"""if output_dir is not None:
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    print(doc2.cats)
"""
#nlp.to_disk("textcat")
test_text="this species is wild and crazy"
doc = nlp(test_text)
print(test_text, doc.cats)
print("testing")
#nlp = nlp.from_disk("textcat")
predicated = []
for test_text, text_cat in zip(test_documents, test_cats):
    doc = nlp(test_text)
    #print(test_text)
    print(text_cat, doc.cats)
    if doc.cats["SUITABLE"] > doc.cats["UNSUITABLE"]:
        predicated.append(1)
    else:
        predicated.append(0)

print("Accuracy Score: {}".format(accuracy_score(y_test, predicated)))
print('Testing F1 score: {}'.format(f1_score(y_test, predicated, average='weighted')))
