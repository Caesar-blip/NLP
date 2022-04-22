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

def main():
    print("Intro to NLP 2022: Assigment 1")
    print("Nils Breeman, Sebastiaan Bye, Julius Wantenaar\n")

    print("PART B. Understanding the task of ...")
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
    plt.title("Lenght baseline for test")
    plt.show()

    analyzer.set_mode("dev")
    ans_dev = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        analyzer.threshold = threshold
        ans_dev[i] = analyzer.length_baseline()[0]
    print(f"Accuracy on dev {max(ans_dev)}, Accuracy on test {max(ans_test)}")

    plt.plot(thresholds, ans_dev)
    plt.title("Lenght baseline for dev")
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
    plt.show()






main()
