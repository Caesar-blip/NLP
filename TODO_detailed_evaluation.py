from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import spacy
import sys
import numpy as np
import pandas as pd
from collections import Counter
import textacy
from wordfreq import word_frequency
import scipy
from scipy import stats
from baselines import Baseliner
import matplotlib.pyplot as plt

def binarize(labels):
    output = []
    for label in labels:
        if label == "N":
            output.append(0)
        elif label == "C":
            output.append(1)
    return output

def get_precision(predicts, labels, target):
    right = 0
    wrong = 0
    i=0
    for predict in predicts:
        if predict == target:
            if predict == labels[i]:
                right +=1
            else:
                wrong +=1
        i+=1
    if (right+wrong) == 0:
        return 0
    return right/(wrong+right)



def main():
    print("Intro to NLP 2022: Assigment 1")
    print("Nils Breeman, Sebastiaan Bye, Julius Wantenaar\n")

    print("PART C. Modeling the task")
    print("\nQuestion 10. Baselines")

    header_names = ['word', 'gold', 'predict']
    data = pd.read_csv('experiments/base_model/model_output.tsv', encoding = 'latin-1', sep='\t', header = 0, names = header_names)
    data = data.dropna()
    
    
    #binarize
    data['gold'] = binarize(list(data['gold']))
    data['predict'] = binarize(list(data['predict']))
    
    target=0
    print("Class N")
    print(f"Precision: {precision_score(data['gold'], data['predict'], pos_label=target)}")
    print(f"Recall : {recall_score(data['gold'], data['predict'], pos_label=target)}")
    print(f"F1: {f1_score(data['gold'], data['predict'], pos_label=target)}")
    average="weighted"
    print(f1_score(data['gold'], data['predict'], pos_label=target,average=average))
    
    print()
    print("Class C")
    target = 1
    print(f"Precision: {precision_score(data['gold'], data['predict'], pos_label=target)}")
    print(f"Recall : {recall_score(data['gold'], data['predict'], pos_label=target)}")
    print(f"F1: {f1_score(data['gold'], data['predict'], pos_label=target)}")
    average="weighted"
    f1 = f1_score(data['gold'], data['predict'], pos_label=target,average=average)
    print(f1)
    
    
main()
