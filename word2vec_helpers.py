from gensim.models import Word2Vec
import gensim.downloader

import numpy as np

def Word2Vec_features(X,model_w2v,preTrain):
    N = len(X)
    X_vec = np.empty((N,model_w2v.vector_size))
    for i in range(N):
        # Constructing sentence feature
        value_iter = np.zeros((model_w2v.vector_size,))
        for word in X[i]:
            try:
                if preTrain == None:
                    word_vec = model_w2v.wv[word]
                else:
                    word_vec = model_w2v[word]
                value_iter += np.array(word_vec) / len(X[i])
            except:
                print('Issue for: X=',i,'with word "', word,'".')
                print('Word ignored in feature construction.')
        X_vec[i,:] = value_iter
    return X_vec

def getWord2Vec(X,preTrain = None):
    '''
    preTrain - string, Allows you to select pretrained model
    '''
    N = len(X)
    if preTrain == 'twitter-25':
        model_w2v = gensim.downloader.load('glove-twitter-25')
    elif preTrain == 'twitter-50':
        model_w2v = gensim.downloader.load('glove-twitter-50')
    elif preTrain == 'gigaword-100':
        model_w2v = gensim.downloader.load('glove-wiki-gigaword-100')
    else:
        vector_size = 100
        N = len(X)
        model_w2v = Word2Vec(X,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4)
        model_w2v.train(X,total_examples=N,epochs= 5)

    return Word2Vec_features(X,model_w2v,preTrain)
