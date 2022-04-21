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
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/preprocessed', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")


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

def analyze_model(directory):
    header_names = ['word', 'gold', 'predict']
    data = pd.read_csv(os.path.join(directory, "model_output.tsv"), encoding = 'latin-1', sep='\t', header = 0, names = header_names)
    data = data.dropna()
    print(f"model: {directory}")

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
    return(f1)

def main():
    print("Intro to NLP 2022: Assigment 1")
    print("Nils Breeman, Sebastiaan Bye, Julius Wantenaar\n")
    print("PART C. Modeling the task")

#    args = parser.parse_args()
#    if  args.model_dir == 'experiments/base_model':
#        print("\nQuestion 10. Baselines")
#    else:
#        print("\nQuestion 14: Hyperparameter changing")
#    directories = args.model_dir.split(" ")
#    for directory in directories:
#        analyze_model(directory)
    
    args = parser.parse_args()
    scores = []
    print("Question 10.")
    scores.append(analyze_model('experiments/base_model'))
    print("\nQuestion 14.")
    print("\n20 hidden nodes")
    scores.append(analyze_model('experiments/hl_exp/hl20/'))
    print("\n80 hidden nodes")
    scores.append(analyze_model('experiments/hl_exp/hl80/'))
    print("\n150 hidden nodes")
    scores.append(analyze_model('experiments/hl_exp/hl150'))

    nodes = [20, 50, 80, 150]
    
    plt.plot(nodes, scores)
    plt.title("F1 as a function of nodes in the hidden layer")
    plt.xlabel("Amount of nodes")
    plt.ylabel("F1 scores")
    plt.show()





main()
