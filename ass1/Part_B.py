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

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def get_precision(predicts, labels, target):
    right = 0
    wrong = 0
    i=0
    for predict in labels:
        if predict == target:
            if predict == predicts[i]:
                right +=1
            else:
                wrong +=1
        i+=1
    if (right+wrong) == 0:
        return 0
    return right/(wrong+right)

def get_recall(predicts, labels, target):
    right=0
    wrong=0
    i=0
    for predict in predicts:
        if predict == target:
            if predict == labels[i]:
                right+=1
            else:
                wrong+=1
        i+=1
    if (right+wrong) == 0:
        return 0
    return right/(wrong+right)

def get_f1(predicts, labels, target):
    recall = get_recall(predicts, labels, target)
    precision = get_precision(predicts, labels, target)
    if (precision+recall) == 0:
        return 0
    return 2*(precision*recall)/(precision+recall)

def binarize(labels):
    output = []
    for label in labels:
        if label == "N":
            output.append(0)
        elif label == "C":
            output.append(1)
    return output

def binarize_labels(labels):
    output = []
    for sent in labels:
        for label in sent:
            if label == "C":
                output.append(1)
            elif label == "N":
                output.append(0)
    return output

def sk_f1(gold, predict, pos_label=1, average="binary"):
    gold = binarize_labels(gold)
    predict = binarize_labels(predict)
    return f1_score(gold, predict, pos_label=pos_label,average=average)

def sk_precision(gold, predict, pos_label=1, average="binary"):
    gold = binarize_labels(gold)
    predict = binarize_labels(predict)
    return precision_score(gold, predict, pos_label=pos_label,average=average)

def sk_recall(gold, predict, pos_label=1, average="binary"):
    gold = binarize_labels(gold)
    predict = binarize_labels(predict)
    return recall_score(gold, predict, pos_label=pos_label,average=average)

def main():
    print("Intro to NLP 2022: Assigment 1")
    print("Nils Breeman, Sebastiaan Bye, Julius Wantenaar\n")

    print("PART B. Understanding the task of ...")

    print("\nQuesiton 7. Basic Statistics")

    nlp = spacy.load("en_core_web_sm")
    header_names = ['HIT ID', 'Sentence', 'Start word', 'End word', 'Target word', 'Native', 'Non-native', 'Difficult native', 'Difficult non', 'Binary', 'Prob']
    tsv_data = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t', header = 0, names = header_names)
    data = tsv_data[['Target word','Binary','Prob']]
    print(data['Binary'].value_counts())
    print(data['Prob'].describe())

    max_tokens = 0
    more_than_one_token = 0
    for word in data['Target word']:
        doc = nlp(word)
        if len(doc) > 1:
            more_than_one_token += 1
        if len(doc) > max_tokens:
            max_tokens = len(doc)
    print(f'Number of instances with more than one token: {more_than_one_token}')
    print(f'Max number of tokens per instance: {max_tokens}')

    print()
    print("\nQuestion 8. Explore linguistic characeteristics")
    temp_data = tsv_data.loc[(tsv_data["Difficult native"] >= 1) | (tsv_data["Difficult non"] >= 1)]
    clean = temp_data
    lengths = []
    
    for i in range(len(temp_data)):
        doc = nlp(clean["Target word"].iloc[i])
        if len(doc) == 1:
            lengths.append(1)
        else:
            lengths.append(0)
    clean['lengths'] = lengths
    clean = clean.loc[(clean['lengths'] == 1)]

    wordcor = []
    postags = []
    
    for i in range(len(clean)):
        for token in nlp(clean['Sentence'].iloc[i]):
            if token.text ==clean['Target word'].iloc[i]:
                pos=token.pos_
                postags.append([pos, clean['Prob'].iloc[i]])
                break
                                      
        wordcor.append([len(clean['Target word'].iloc[i]), 
        word_frequency(clean['Target word'].iloc[i], 'en', minimum = 0.0),
            clean['Prob'].iloc[i]])

    # x is word length, y is probabilistic complexity
    word_length = np.array(wordcor).T[0]
    word_freq = np.array(wordcor).T[1]
    prob_complex = np.array(wordcor).T[2]
    
    print(f"Pearson Correlation length vs complexity: {scipy.stats.pearsonr(word_length, prob_complex)[0]}, p = {scipy.stats.pearsonr(word_length, prob_complex)[1]}")
    print(f"Pearson Correlation frequency vs complexity: {scipy.stats.pearsonr(word_freq, prob_complex)[0]}, p = {scipy.stats.pearsonr(word_freq, prob_complex)[1]}")
    
    # Make some plots
    plt.scatter(word_length, prob_complex)
    plt.title("Plot between words length and  probabilistic complexity")
    plt.xlabel("Word length")
    plt.ylabel("Probabilistic Complexity")
    plt.xticks([0, 5, 10, 15, 20])
    plt.show()

    plt.scatter(word_freq, prob_complex)
    plt.title("Plot between words frequency and  probabilistic complexity")
    plt.xlabel("Word Frequency")
    plt.ylabel("Probabilistic Complexity")
    plt.show()

    plt.scatter(np.array(postags).T[0], np.array(postags).T[1].astype('float'))
    plt.title("Plot between POS tag and probabilistic complexity")
    plt.xlabel("POS Tag")
    plt.ylabel("Probabilistic Complexity")
    plt.show()



    print("\nQuestion 10. Baselines")
    nlp = spacy.load("en_core_web_sm")
    header_names = ['HIT ID', 'Sentence', 'Start word', 'End word', 'Target word', 'Native', 'Non-native', 'Difficult native',
             'Difficult non', 'Binary', 'Prob']
    tsv_data = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t', header = 0, names = header_names)
    analyzer = Baseliner()
    runs = 1000

    # Majority
    print("Majority Baseline")

    analyzer.set_mode("test")
    maj_test_ans = np.zeros(runs)
    for i in range(runs):
        maj_test_ans = analyzer.majority_baseline()[0]
    
    analyzer.set_mode("dev")
    maj_dev_ans = np.zeros(runs)
    for i in range(runs):
        maj_dev_ans[i] = analyzer.majority_baseline()[0]
    print(f"Accuracy on dev {np.mean(maj_dev_ans)}, Accuracy on test {np.mean(maj_test_ans)}")
    
    # Random
    print("\nRandom Baseline")
    analyzer.set_mode("test")
    maj_test_ans = np.zeros(runs)
    for i in range(runs):
        maj_test_ans = analyzer.random_baseline()[0]
    
    analyzer.set_mode("dev")
    maj_dev_ans = np.zeros(runs)
    for i in range(runs):
        maj_dev_ans[i] = analyzer.random_baseline()[0]
    print(f"Accuracy on dev {np.mean(maj_dev_ans)}, Accuracy on test {np.mean(maj_test_ans)}")

    # Length
    print("\nLength Baseline")
    thresholds = [2,3,4,5,6,7,8,9,10,11,12]
    analyzer.set_mode("test")

    ans_test = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        analyzer.threshold = threshold
        ans_test[i] = analyzer.length_baseline()[0]

    plt.plot(thresholds, ans_test)
    plt.title("Length baseline for test")
    plt.xlabel("Thresholds")
    plt.ylabel("Accurarcy")
    plt.show()

    analyzer.set_mode("dev")
    ans_dev = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        analyzer.threshold = threshold
        ans_dev[i] = analyzer.length_baseline()[0]
    print(f"Accuracy on dev {max(ans_dev)}, Accuracy on test {max(ans_test)}")

    plt.plot(thresholds, ans_dev)
    plt.title("Length baseline for dev")
    plt.xlabel("Thresholds")
    plt.ylabel("Accurarcy")
    plt.show()

    # Frequency
    print("\nFrequency baseline")
    analyzer.set_mode("test")
    
    # check for different thresholds
    thresholds = [10e-8, 10e-7, 10e-6, 10e-5,10e-4, 10e-3, 10e-2, 10e-1, 10e-0]

    ans_test = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        analyzer.threshold = threshold
        ans_test[i] = analyzer.frequency_baseline()[0]
    
    plt.xscale("log")
    plt.plot(thresholds, ans_test)
    plt.title("Frequency baseline for test")
    plt.xlabel("Frequency Thresholds")
    plt.ylabel("Accurarcy")
    plt.show()

    analyzer.set_mode("dev")
    ans_dev = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        analyzer.threshold = threshold
        ans_dev[i] = analyzer.frequency_baseline()[0]
    print(f"Accuracy on dev {max(ans_dev)}, Accuracy on test {max(ans_test)}")

    plt.xscale("log")
    plt.plot(thresholds, ans_dev)
    plt.title("Frequency baseline for dev")
    plt.xlabel("Frequency Thresholds")
    plt.ylabel("Accurarcy")
    plt.show()

    print(f"Precision, Accurarcy and F1")
    analyzer = Baseliner()
    analyzer.set_mode("test")
    
    print("\n Results Majority Baseline")
    results = analyzer.majority_baseline()
    target = 1
    
    print("Class C")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    
    target = 0
    print("\nClass N")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    
    print(f'\nWeighted:{sk_f1(results[1], analyzer.testlabels,pos_label=target, average="weighted")}')

    print("\n\n Results Random Baseline")
    results = analyzer.random_baseline()
    target = 1

    print("Class C")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')

    target = 0
    print("\nClass N")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')

    print(f'\nWeighted:{sk_f1(results[1], analyzer.testlabels,pos_label=target, average="weighted")}')

    print("\n\n Length baseline")
    analyzer.threshold=7
    results = analyzer.length_baseline()
    target = 1

    target = 1

    print("Class C")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')

    target = 0
    print("\nClass N")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')

    print(f'\nWeighted:{sk_f1(results[1], analyzer.testlabels,pos_label=target, average="weighted")}')


    print("\n\n Results Frequency Baseline")
    analyzer.threshold=10e-6
    results = analyzer.frequency_baseline()

    target = 1
    print("Class C")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')

    target = 0
    print("\nClass N")
    print(f'Precision: {sk_precision(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_precision(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'Recall: {sk_recall(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_recall(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')
    print(f'F1 {sk_f1(results[1], analyzer.testlabels,pos_label=target, average="binary")}')
    print(f'{get_f1(binarize_labels(results[1]), binarize_labels(analyzer.testlabels), target)}')

    print(f'\nWeighted:{sk_f1(results[1], analyzer.testlabels,pos_label=target, average="weighted")}')


main()
