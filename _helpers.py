import textdistance
import re
import pandas as pd
import numpy as np
from collections import Counter


def autocorrection(input_word, vocab_data = pd.read_csv("snips/merged_GT_data.csv")):
    """return autocorrected input word"""


    words = list(vocab_data["transcript"].explode().str.split(" ").explode())
    V = set(words) #create a set of words present in a dictionary

    word_freq = {}
    word_freq = Counter(words) #word freq
    probs = {}
    Total = sum(word_freq.values())
    for k in word_freq.keys():
        probs[k] = word_freq[k]/Total #calculate word probability


    input_word = input_word.lower()
    if input_word in V:
        return input_word
    else:
        sim = [1 - (textdistance.Jaccard(qval = 2).distance(v, input_word)) for v in word_freq.keys()] #find the most similar word in set based in word distance
        auto_df = pd.DataFrame.from_dict(probs, orient = "index").reset_index()
        auto_df = auto_df.rename(columns = {"index":"Word", 0: "Prob"})
        auto_df["Similarity"] = sim 
        output = auto_df.sort_values(["Similarity", "Prob"], ascending = False).reset_index()["Word"][0] #sort based in similarity and probability 
        return output


def indx2action(y_num):
    """index to action"""
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
        elif y_num[i] == 4:
            y.append('SetLightBrightness')
        elif y_num[i] == 5:
            y.append('SetLightColor')
        else:
            y.append('No class')
    return y

#label data based in keywords
def label_data(df):
    """return a dataframe labeled based on keywords"""
    
    y = []
    y_raw = df["keywords"]
    N = len(y_raw)
    y_num = np.empty(N)
    i = 0
    for line in y_raw:
        if 'turn off' in line:
            y.append('SwitchLightOff')
            y_num[i] = 0
        elif 'turn on' in line:
            y.append('SwitchLightOn')
            y_num[i] = 1
        elif 'increase' in line:
            y.append('IncreaseBrightness')
            y_num[i] = 2
        elif 'decrease' in line:
            y.append('DecreaseBrightness')
            y_num[i] = 3
        else:
            y.append('No class')
            y_num[i] = 6
        i += 1
    df["user_action"] = y
    df["user_action_num"] = y_num
    return df

def action2index(action):
    """return the index associated with the action"""
    if action == 'SwitchLightOff':
        return 0
    elif action == 'SwitchLightOn':
        return 1
    elif action == 'IncreaseBrightness':
        return 2
    elif action == 'DecreaseBrightness':
        return 3
    elif action == 'SetLightBrightness':
        return 4
    elif action == "SetLightColor":
        return 5
    elif action == 'No class':
        return 4

def padding_func(x, _max):
    """returns a pad version of x with _max as len"""
    return np.array(x + [0]*(_max-len(x)))