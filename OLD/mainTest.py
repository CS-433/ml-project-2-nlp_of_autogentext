# Downloads
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# Import 
from gensim.test.utils import common_texts
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

# Heavy stuff
import gensim.downloader

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

import numpy as np
import pandas as pd
import json

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
        X_in = df.values[:,3]
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
    elif type(rawData[0]) is list:
        '''
        Process CSV
        '''
        for sent in rawData[0]:
            words = [''.join(e for e in word if e.isalnum()).lower() for word in sent]     # Remove special characters and make lowercase
            if process:
                words = [lemmatizer.lemmatize(word) for word in words if word not in ( set(stopwords.words('english'))-important_words)]
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
    
def appendEntry(X,Y,x,y):
    '''
    might be pointless hehe
    '''
    X.append(x)
    Y.append(y)
    return X,y

def doCrossvalidation(X,y,model,fold):
    CV_results = cross_validate(model,X,y,cv = fold)
    print('Cross-validation score: ',CV_results['test_score'])

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
    # X = getWord2Vec(X_raw,'gigaword-100')
    X = getTFIDF(X_raw)

    # Train model
    n = 1
    F1_mean = np.zeros((n,len(dct_y)))
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(X,y_num,test_size = 0.5)
        cls = LogisticRegression()
        cls.fit(x_train,y_train) 

        # Evaluate performance
        y_pred = cls.predict(x_test)
        score = cls.score(x_test,y_test)
        F1 = f1_score(y_test,y_pred,average=None)
        F1_mean[i,:] = F1
    F1_crossVal = np.mean(F1_mean,axis =0)
    print(score)
    print(dct_y)
    print("F1 crossvalidation for",n,"iterations:",F1_crossVal)

mainTest()  
