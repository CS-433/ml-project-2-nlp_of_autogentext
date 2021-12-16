import numpy as np
import pandas as pd
import json

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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