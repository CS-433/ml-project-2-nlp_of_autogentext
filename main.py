# Downloads
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# Import
from gensim.test.utils import common_texts
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB



import numpy as np
import pandas as pd
import json
import csv

def readJson(fileLocation,process = True):
    '''
    process - boolean, wether or not to preprocess data
    '''
    File = open(fileLocation)
    metaData_in = json.load(File)
    keys = list(metaData_in.keys())
    return doPreprocessing([metaData_in,keys],process)

def readCSV(fileLocation,ASR = False,process = True):
    '''
    ASR - boolean, if true read ASR data instead of groundtruth transcript
    '''
    df = pd.read_csv(fileLocation, sep=',',header=0)
    y_raw = df.values[:,4]
    if ASR:
        X_in = df.values[:,:,3]
        N = len(X_in)
        X_raw = []
        for i in range(N):
            X_raw.append(X_in[i].split(' '))
    else:
        X_in = df.values[:,2]
        N = len(X_in)
        X_raw = []
        for i in range(N):
            X_raw.append(X_in[i].split(' '))
    return doPreprocessing([X_raw,y_raw],process)

def labelCSV(y):
    Y = list(set(y))
    y_num = []
    for action in y:
        y_num.append(Y.index(action))
    return y_num,Y

def indx2action(y_num):
    N = len(y_num)
    y =[]
    for i in range (N):
        if y_num[i] == 0:
            y.append('SwitchLightOff')
        elif y_num[i] == 1:
            y.append('SwitchLightOn')
        elif y_num[i] == 2:
            y.append('IncreaseBrightness')
        elif  y_num[i] == 3:
            y.append('DecreaseBrightness')
        else:
            y.append('No class assigned')
    return y

def labelData(y_raw):
    N = len(y_raw)
    y_num = np.empty(N)
    i = 0
    y = []
    for line in y_raw:
        for word in line:
            if word == 'turn off':
                y.append('SwitchLightOff')
                y_num[i] = 0
                break
            elif word == 'turn on':
                y.append('SwitchLightOn')
                y_num[i] = 1
                break
            elif word == 'increase':
                y.append('IncreaseBrightness')
                y_num[i] = 2
                break
            elif word == 'decrease':
                y.append('DecreaseBrightness')
                y_num[i] = 3
                break
            elif word == 'decrease':
                y.append('DecreaseBrightness')
                y_num[i] = 3
                break
            else:
                y.append('No class')
                y_num[i] = 4
                break
        i += 1
    return y,y_num

def doPreprocessing(rawData,process):
    important_words = {'on','off'}
    lemmatizer = WordNetLemmatizer()
    y = []
    X = []
    if type(rawData[0]) is dict:
        '''
        Process Json
        '''
        N =  len(rawData[0])
        for i in range(N):
            words = rawData[0].get(rawData[1][i]).get('transcript').split(' ')
            if process:
                words = [lemmatizer.lemmatize(word) for word in words if word not in ( set(stopwords.words('english'))-important_words)]
            X.append(words)
            y.append(rawData[0].get(rawData[1][i]).get('keywords'))
    elif type(rawData[0]) is list and process:
        '''
        Process CSV
        '''
        for sent in rawData[0]:
            words = [lemmatizer.lemmatize(word) for word in sent if word not in ( set(stopwords.words('english'))-important_words)]
            X.append(words)
        y = rawData[1]
    else:
        X = rawData[0]
        y = rawData[1]
    return X,y

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

def Word2Vec_features(X,model_w2v):
    N = len(X)
    X_vec = np.empty((N,model_w2v.vector_size))
    for i in range(N):
        # Constructing sentence feature
        value_iter = np.zeros((model_w2v.vector_size,))
        for word in X[i]:
            try:
                value_iter += np.array(model_w2v.wv[word]) / len(X[i])
            except:
                print('Issue for: X=',i,'with word "', word,'".')
                print('Word ignored in feature construction.')
        X_vec[i,:] = value_iter
    return X_vec

def getTFIDF(X):
    dictX = Dictionary(X)
    corpusBOW = [dictX.doc2bow(line) for line in X]
    model = TfidfModel(corpusBOW)
    X_vec = model[corpusBOW]
    return Tfidf_features(X_vec,dictX)

def getWord2Vec(X,preTrain = None):
    '''
    preTrain - string, Allows you to select pretrained model
    '''
    vector_size = 100
    N = len(X)
    model_w2v = Word2Vec(X,
                         vector_size=vector_size,
                         window=5,
                         min_count=1,
                         workers=4)
    if preTrain != None:
        raise ValueError('Not yet implemented')
    else:
        model_w2v.train(X,total_examples=N,epochs= 5)
    return Word2Vec_features(X,model_w2v)


def appendEntry(X,Y,x,y):
    '''
    might be pointless hehe
    '''
    X.append(x)
    Y.append(y)
    return X,y

def main():
    # Raw data to feature embedding
    fileLocation = "Data/openvoc-keyword-spotting-research-datasets/smart-lights/metadata.json"
    fileLocation = 'Data/smart-lights_close_ASR.csv'
    # X_raw,y_raw = readJson(fileLocation,process = False)
    X_raw,y_num = readCSV(fileLocation)
    # y,y_num = labelData(y_raw)
    X = getTFIDF(X_raw)
    # X = getWord2Vec(X_raw)

    # Train model
    x_train, x_test, y_train, y_test = train_test_split(X,y_num,test_size = 0.5)

    # Classifiers below
    cls = LogisticRegression()
    cls.fit(x_train,y_train)

    # Evaluate performance
    y_pred = cls.predict(x_test)
    score = cls.score(x_test,y_test)
    F1 = f1_score(y_test,y_pred,average=None)
    print(score)
    print(F1)

def mainTest():
    fileLocation = 'Data/smart-lights_close_ASR.csv'
    X_raw,y = readCSV(fileLocation,ASR = False,process = False)
    y_num,dct_y= labelCSV(y)
    X = getWord2Vec(X_raw)
    X = getTFIDF(X_raw)

    # Train model
    x_train, x_test, y_train, y_test = train_test_split(X,y_num,test_size = 0.5)
    cls = LogisticRegression()
    cls.fit(x_train,y_train)

    # Evaluate performance
    y_pred = cls.predict(x_test)
    score = cls.score(x_test,y_test)
    F1 = f1_score(y_test,y_pred,average=None)
    print(score)
    print(F1)

    # SVM
    svm_classifier = SVC()
    svm_classifier.fit(x_train,y_train)
    # SVM Evaluation
    y_pred_svm = svm_classifier.predict(x_test)
    score_nvm = svm_classifier.score(x_test,y_test)
    print('SVM Accuracy:',score_nvm)
    F1 = f1_score(y_test,y_pred_svm,average=None)
    print('SVM f1 score:',F1)

    # NN
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=5000)
    mlp.fit(x_train,y_train)
    predict_train = mlp.predict(x_train)
    predict_test = mlp.predict(x_test)
    # Print results for NN
    # Training data matrix
    print(confusion_matrix(y_train,predict_train))
    print(classification_report(y_train,predict_train))
    # Test data matrix
    print(confusion_matrix(y_test,predict_test))
    print(classification_report(y_test,predict_test))

    # Gaussian Bayes
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print("Score of GaussianNB: ", gnb.score(x_test, y_test))
    # Bernoulli Bayes
    bnb = BernoulliNB()
    bnb.fit(x_train, y_train)
    print("Score of BernoulliNB: " , bnb.score(x_test, y_test))

mainTest()

