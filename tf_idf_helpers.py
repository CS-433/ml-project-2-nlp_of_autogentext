from gensim.models import TfidfModel
from gensim.corpora import Dictionary

import numpy as np

def Tfidf_features(X,dct):
    """ 
    Converts gensim format to numpy array
    Input:
    X - TDidfModel vector (N x lenght("sentence"))
    dct - Dictionary object (lenght("unique words"))
    Output:
    X_np - N x length("unique words")
    """
    N_dict = len(dct)
    N_sent = len(X)
    X_np = np.zeros((N_sent,N_dict))
    i = 0
    for list in X:
        for word in list:
            X_np[i, word[0]] = word[1]
        i += 1
    return X_np


def getTFIDF(X):
    dictX = Dictionary(X)
    corpusBOW = [dictX.doc2bow(line) for line in X]
    model = TfidfModel(corpusBOW)
    X_vec = model[corpusBOW]
    return Tfidf_features(X_vec,dictX)