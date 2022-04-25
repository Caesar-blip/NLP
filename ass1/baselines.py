# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

from torch import minimum, threshold
from model.data_loader import DataLoader
from collections import Counter

from model.net import accuracy
import numpy as np
from wordfreq import word_frequency
# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.
class Baseliner():
    def __init__(self):
        self.train_path = "data/preprocessed/train/"
        self.dev_path = "data/preprocessed/val/"
        self.test_path = "data/preprocessed/test/"
        self.threshold = 6
        # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
        with open(self.train_path + "sentences.txt", encoding = "utf8") as sent_file:
            self.train_sentences = sent_file.readlines()
        with open(self.train_path + "labels.txt", encoding = "utf8") as label_file:
            self.train_labels = label_file.readlines()
        with open(self.dev_path + "sentences.txt", encoding = "utf8") as dev_file:
            self.dev_sentences = dev_file.readlines()
        with open(self.dev_path + "labels.txt", encoding = "utf8") as dev_label_file:
            self.dev_labels = dev_label_file.readlines()

        with open(self.test_path + "sentences.txt", encoding = "utf8") as testfile:
            self.testinput = testfile.readlines()
        with open(self.test_path + "labels.txt") as test_label_file:
            self.testlabels = test_label_file.readlines()

        self.mode = "test"
        if self.mode == "test":
            self.input = self.testinput
            self.labels = self.testlabels
        else:
            self.input = self.dev_sentences
            self.labels = self.dev_labels

    def set_mode(self, new_mode):
        self.mode = new_mode
        assert self.mode == "test" or self.mode == "dev", "mode should either be dev or test"

        if self.mode == "test":
            self.input = self.testinput
            self.labels = self.testlabels
        else:
            self.input = self.dev_sentences
            self.labels = self.dev_labels

    def majority_baseline(self):
        predictions = []
        # find majority label
        majority_class = Counter(' '.join(self.train_labels)).most_common(2)[1][0]

        predictions = []
        for instance in self.input:
            sent_predicts = []
            tokens = instance.split(" ")
            for token in tokens:
                sent_predicts.append(majority_class)
            predictions.append(sent_predicts)
        # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
        correct = 0
        j = 0
        for sent in self.labels:
            i = 0
            labels = sent.split(" ")
            for char in labels:
                if char == predictions[j][i]:
                    correct+=1
                i+=1
            j+=1
        accuracy = correct/sum([len(sent) for sent in predictions])
        return accuracy, predictions


    def random_baseline(self):
        predictions = []
        for instance in self.input:
            sent_predicts = []
            tokens = instance.split(" ")
            for token in tokens:
                if np.random.random() <= 0.5:
                    sent_predicts.append("N")
                else:
                    sent_predicts.append("C")
            predictions.append(sent_predicts)
        # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
        correct = 0
        j = 0
        for sent in self.labels:
            i = 0
            labels = sent.split(" ")
            for char in labels:
                if char == predictions[j][i]:
                    correct+=1
                i+=1
            j+=1
        accuracy = correct/sum([len(sent) for sent in predictions])
        return accuracy, predictions
    
    
    def frequency_baseline(self):
        predictions = []
        for instance in self.input:
            sent_predicts = []
            tokens = instance.split(" ")
            for token in tokens:
                if word_frequency(token, 'en', minimum=0.0) < self.threshold:
                    sent_predicts.append("C")
                else:
                    sent_predicts.append("N")
            
            predictions.append(sent_predicts)
    
        # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
        correct = 0
        j = 0
        for sent in self.labels:
            i = 0
            labels = sent.split(" ")
            for char in labels:
                if char == predictions[j][i]:
                    correct+=1
                i+=1
            j+=1
    
        accuracy = correct/sum([len(sent) for sent in predictions])
        return accuracy, predictions


    def length_baseline(self):
        predictions = []
        for instance in self.input:
            sent_predicts = []
            tokens = instance.split(" ")
            for token in tokens:
                if len(token) > self.threshold:
                    sent_predicts.append("C")
                else:
                    sent_predicts.append("N")
            
            predictions.append(sent_predicts)
    
        # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
        correct = 0
        j = 0
        for sent in self.labels:
            i = 0
            labels = sent.split(" ")
            for char in labels:
                if char == predictions[j][i]:
                    correct+=1
                i+=1
            j+=1
    
        accuracy = correct/sum([len(sent) for sent in predictions])
        return accuracy, predictions

def majority_baseline(train_labels, testinput, testlabels):
    predictions = []
    # find majority label
    majority_class = Counter(' '.join(train_labels)).most_common(2)[1][0]

    predictions = []
    for instance in testinput:
        sent_predicts = []
        tokens = instance.split(" ")
        for token in tokens:
            sent_predicts.append(majority_class)
        
        predictions.append(sent_predicts)

    # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
    correct = 0
    j = 0
    for sent in testlabels:
        i = 0
        labels = sent.split(" ")
        for char in labels:
            if char == predictions[j][i]:
                correct+=1
            i+=1
        j+=1

    accuracy = correct/sum([len(sent) for sent in predictions])
    return accuracy, predictions


def random_baseline(testinput, testlabels):
    predictions = []

    predictions = []
    for instance in testinput:
        sent_predicts = []
        tokens = instance.split(" ")
        for token in tokens:
            if np.random.random() <= 0.5:
                sent_predicts.append("N")
            else:
                sent_predicts.append("C")
        
        predictions.append(sent_predicts)

    # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
    correct = 0
    j = 0
    for sent in testlabels:
        i = 0
        labels = sent.split(" ")
        for char in labels:
            if char == predictions[j][i]:
                correct+=1
            i+=1
        j+=1

    accuracy = correct/sum([len(sent) for sent in predictions])
    return accuracy, predictions


def frequency_baseline(threshold, testinput, testlabels):
    predictions = []

    predictions = []
    for instance in testinput:
        sent_predicts = []
        tokens = instance.split(" ")
        for token in tokens:
            if word_frequency(token, 'en', minimum=0.0) > threshold:
                sent_predicts.append("C")
            else:
                sent_predicts.append("N")
        
        predictions.append(sent_predicts)

    # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
    correct = 0
    j = 0
    for sent in testlabels:
        i = 0
        labels = sent.split(" ")
        for char in labels:
            if char == predictions[j][i]:
                correct+=1
            i+=1
        j+=1

    accuracy = correct/sum([len(sent) for sent in predictions])
    return accuracy, predictions


def length_baseline(threshold, testinput, testlabels):
    predictions = []

    predictions = []
    for instance in testinput:
        sent_predicts = []
        tokens = instance.split(" ")
        for token in tokens:
            if len(token) > threshold:
                sent_predicts.append("C")
            else:
                sent_predicts.append("N")
        
        predictions.append(sent_predicts)

    # calculate accuracy for the test input, accuracy = correct/(correct+incorrect)
    correct = 0
    j = 0
    for sent in testlabels:
        i = 0
        labels = sent.split(" ")
        for char in labels:
            if char == predictions[j][i]:
                correct+=1
            i+=1
        j+=1

    accuracy = correct/sum([len(sent) for sent in predictions])
    return accuracy, predictions


if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"
    threshold = 6

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.

    with open(train_path + "sentences.txt", encoding = "utf8") as sent_file:
        train_sentences = sent_file.readlines()

    with open(train_path + "labels.txt", encoding = "utf8") as label_file:
        train_labels = label_file.readlines()

    with open(dev_path + "sentences.txt", encoding = "utf8") as dev_file:
        dev_sentences = dev_file.readlines()
    with open(train_path + "labels.txt", encoding = "utf8") as dev_label_file:
        dev_labels = dev_label_file.readlines()
    
    with open(test_path + "sentences.txt", encoding = "utf8") as testfile:
        testinput = testfile.readlines()
    with open(test_path + "labels.txt") as test_label_file:
        testlabels = test_label_file.readlines()
    

    majority_accuracy, majority_predictions = majority_baseline(train_labels, testinput, testlabels)
    random_accuracy, random_predictions = random_baseline(testinput, testlabels)
    frequency_accuracy, frequency_predictions = frequency_baseline(threshold, testinput, testlabels)
    length_accuracy, length_predictions = length_baseline(threshold, testinput, testlabels)

    # TODO: output the predictions in a suitable way so that you can evaluate them
    print(length_accuracy, length_predictions)
