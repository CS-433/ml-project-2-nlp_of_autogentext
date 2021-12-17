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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB

import nlpaug.augmenter.word as naw

import numpy as np
import pandas as pd
import json
import csv
import random
import gzip
import shutil
import os
import wget

def readJson(fileLocation,process = True):
    '''
    process - boolean, wether or not to preprocess data
    '''
    File = open(fileLocation)
    metaData_in = json.load(File)
    keys = list(metaData_in.keys())
    return doPreprocessing([metaData_in,keys],process)

def readCSV(fileLocation,ASR = False,process = True, augment= False, aug_rows=400):
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
    return doPreprocessing([X_raw,y_raw],process, augment, aug_rows)

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

def doPreprocessing(rawData,process, augment, aug_rows):
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
    if augment:
        model_path = download_naw_model()
        augmented_X = text_augmenter_word_embedder(X, model_path, aug_rows)
        return augmented_X, y
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

def download_naw_model():
    gn_vec_path = "GoogleNews-vectors-negative300.bin"
    if not os.path.exists("GoogleNews-vectors-negative300.bin"):
        if not os.path.exists("../Ch3/GoogleNews-vectors-negative300.bin"):
            # Downloading the reqired model
            if not os.path.exists("../Ch3/GoogleNews-vectors-negative300.bin.gz"):
                if not os.path.exists("GoogleNews-vectors-negative300.bin.gz"):
                    wget.download("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz")
                gn_vec_zip_path = "GoogleNews-vectors-negative300.bin.gz"
            else:
                gn_vec_zip_path = "../Ch3/GoogleNews-vectors-negative300.bin.gz"
            # Extracting the required model
            with gzip.open(gn_vec_zip_path, 'rb') as f_in:
                with open(gn_vec_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            gn_vec_path = "../Ch3/" + gn_vec_path

    print(f"Model at {gn_vec_path}")
    return f"./{gn_vec_path}"

def text_augmenter_word_embedder(texts, model_path, rows):
    # model_type: word2vec, glove or fasttext
    aug = naw.WordEmbsAug(model_type='word2vec', model_path=model_path, action="substitute")
    for i in range(0, 400):
        augmented_text = aug.augment(' '.join(texts[i]))
        texts[i] = augmented_text.split()
    return texts

def random_deletion(sentence, p=0.3):
    words = sentence.split ()
    n = len (words)
    if n == 1: # return if single word
        return words
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words))
    #print (remaining)
    if len(remaining) == 0: # if not left, sample a random word
        return ' '.join ([random.choice(words)])
    else:
        return ' '.join (remaining)


def mainTest(augment=False):
    fileLocation = 'Data/smart-lights_close_ASR.csv'
    augmented_X_raw = []
    aug_rows = 400
    X_raw,y = readCSV(fileLocation,ASR = True,process = False, augment= True, aug_rows=aug_rows)
    y_num,dct_y= labelCSV(y)
    #X = getWord2Vec(X_raw,'gigaword-100')
    X = getTFIDF(X_raw)
    print("X!")
    print(X.shape)
    print('AUUGMENTEd')
    print(X[0:aug_rows])

    # Train model
    n = 10
    F1_mean = np.zeros((n,len(dct_y)))
    classifiers = {'LogisticRegression': LogisticRegression(), 'SVM': SVC(), 'GaussianNB': GaussianNB(), 'MLPClassifier': MLPClassifier(hidden_layer_sizes=(400, 100), activation='relu', solver='adam', max_iter=1000) }
    classifier_score = dict.fromkeys(classifiers.keys(),[])
    for key, value in classifiers.items():
        for i in range(n):
            x_train, x_test, y_train, y_test = train_test_split(X[aug_rows:],y_num[aug_rows:],test_size = 0.5)
            np.concatenate((x_train, X[0:aug_rows]))
            np.concatenate((y_train, y_num[0:aug_rows]))
            classifier = value
            classifier.fit(x_train,y_train)

            # Evaluate performance
            y_pred = classifier.predict(x_test)
            #score = classifier.score(x_test,y_test)
            #print(confusion_matrix(y_test,y_pred))
            F1 = f1_score(y_test,y_pred,average=None)
            F1_mean[i,:] = F1
        F1_crossVal = np.mean(F1_mean,axis =0)
        print(dct_y)
        print("F1 crossvalidation for",n,"iterations:",F1_crossVal)
        print(key)
        print('score: ‰f' % F1_crossVal.mean())
        classifier_score[key] = F1_crossVal.mean()

    print(classifier_score)
mainTest()


"""
if not os.path.exists("spelling_en.txt"):
    wget.download("https://raw.githubusercontent.com/makcedward/nlpaug/5238e0be734841b69651d2043df535d78a8cc594/nlpaug/res/word/spelling/spelling_en.txt")
else:
    print("File already exists")
    
    
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') 
    
"""
