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
#    print("PART A: Linguistic analysis using spaCy")
#    print(f"Version of Spacy used for development 3.2.4 Current verion used: {spacy.__version__}")
#    print(f"Version of Python used for development 3.8.13 Current version used: {sys.version}")
#
#    # Question 1. Tokenization
#    print("\nQuestion 1. Tokenization")
#    nlp = spacy.load("en_core_web_sm")
#
#    with open ("data/preprocessed/train/sentences.txt", encoding = "utf8") as text:
#        data = text.readlines()
#        string = ""
#        for i in range(len(data)):
#            string += str(data[i]).replace("\n", " ").replace("\\", "")
#    doc = nlp(string)
#
#    # Number of tokens
#    count = 0
#    for token in doc:
#        count+=1
#    print(f"Number of tokens {count}")
#
#    # Number of types
#    count = 0
#    unique = []
#    
#    for words in doc:
#        unique.append(words.text)
#            
#    unique = np.unique(unique)    
#    print(f"The number of types {len(unique)}")
#    
#    # Number of words
#    skip = ["PUNCT"]
#    word_list = []
#    for token in doc:
#        if not token.is_punct:
#            word_list.append(token.text)
#    print(f"Number of words: {len(word_list)}")
#
#    # Average of words per sentence
#    wordcount = len(word_list)
#    sentcount = len(list(doc.sents))
#    print(f"The average words per sentence is {wordcount/sentcount}")
#    
#    word_lengths = []
#    for i in word_list:
#        word_lengths.append(len(i))
#    print(f"The average word length is {np.mean(word_lengths)} +/- {np.std(word_lengths)}")
#
#
#    # Question 2. 
#    print("\nQuestion 2. Word Classes")
#
#    tokens = [token.tag_ for token in doc]
#    tokens = np.array(tokens)
#    tokens = np.unique(tokens)
#    token_count = dict.fromkeys(tokens, 0)
#    for token in doc:
#        token_count[token.tag_] += 1
#    
#    my_keys = sorted(token_count, key=token_count.get, reverse=True)[:10]
#    uberdict = dict.fromkeys(my_keys, 0)
#
#    for i in my_keys:
#        # get universal POS applicable to finegrained POS
#        uniPOS = [token.pos_
#            for token in doc if token.tag_ == i]
#        uniPOS = list(set(uniPOS))
#        # get all words
#        all_words = [token.text
#                for token in doc if token.tag_ ==i]
#         # get count of words for finegrained POS token
#        fineg_count = len(all_words)
#        # create dict with all words existing for this finegrained POS token
#        words = np.array(all_words)
#        words = np.unique(words)
#        word_dict = dict.fromkeys(words, 0) 
#        # count number of occurences of word 
#        for tok in all_words:
#            word_dict[tok] += 1
#            # get most common and least common words
#            keys_freqtokens = sorted(word_dict, key=word_dict.get, reverse=True)[:3]
#            keys_unfreqtokens = sorted(word_dict, key=word_dict.get)[:1]
#            uberdict[i] = [uniPOS, fineg_count, keys_freqtokens, keys_unfreqtokens]
#
#    total_tags = 0
#    for i in uberdict.values():
#        total_tags += i[1]
#    
#        for i in uberdict.keys():
#            uberdict[i].append(uberdict[i][1]/total_tags)
#    df = pd.DataFrame.from_dict(uberdict,orient="index")
#    print(df)
#
#
#    # Question 3.
#    print("\nQuestion 3. N-Grams")
#
#
#    print("\n Question 4. Lemmatization")
#
#
#    print("\n Question 5. Named Entity Recognition")
#
## Create sentence object and an array for entity information
#    doc_sent = doc.sents
#    array = np.zeros(len(list(doc.sents))).tolist()
#
## Loop over sentences save entities in array
#    counter = 0
#    first_five = []
#    for sent in doc.sents:
#        first_five.append(sent)
#        temp_entity = []
#        sent_text = nlp(str(sent))
#
#        for ent in sent_text.ents:
#            temp_entity.append([ent.text, ent.label_])
#        array[counter] = temp_entity
#        counter += 1
#    
#    # Get entities and entity labels back from array
#    named_entities = 0
#    entity_labels = []
#    
#    for i in range(len(array)):
#        temp_array = np.array(array[i]).T
#        if temp_array.size != 0:
#            named_entities += len(temp_array[0]) + 1
#            entity_labels.append(temp_array[1].tolist())
#                                        
#    print(f"Number of named entities: {named_entities}")    
#    print(f"Number of different entity labels: {len(set(np.hstack(entity_labels).tolist()))}")
#    
#    
#    # Analyse first five sentences
#    for i in range(0,5):
#        print(f"Sentence {i+1}:")
#        print(first_five[i])
#        if np.array(array[i]).T.size != 0:
#            print(f"Entities: {np.array(array[i]).T[0]}")
#            print(f"Labels: {np.array(array[i]).T[1]}")
#                                    
#        print()
#    
    print("\n")
    print("\n PART B. Understanding the task of ...")
    print("\nQuesiton 7. Basic Statistics")

    nlp = spacy.load("en_core_web_sm")
    header_names = ['HIT ID', 'Sentence', 'Start word', 'End word', 'Target word', 'Native', 'Non-native', 'Difficult native',
             'Difficult non', 'Binary', 'Prob']
    tsv_data = pd.read_csv('data/original/english/WikiNews_Train.tsv', sep='\t', header = 0, names = header_names)
    data = tsv_data[['Target word','Binary','Prob']]
    print(data['Binary'].value_counts())
    print(data['Prob'].describe())

#    max_tokens = 0
#    more_than_one_token = 0
#    for word in data['Target word']:
#        doc = nlp(word)
#        if len(doc) > 1:
#            more_than_one_token += 1
#        if len(doc) > max_tokens:
#            max_tokens = len(doc)
#    print(f'Number of instances with more than one token: {more_than_one_token}')
#    print(f'Max number of tokens per instance: {max_tokens}')

    print("\nQuestion 8. Explore linguistic characeteristics")
    temp_data = tsv_data.loc[(tsv_data["Difficult native"] >= 1) | (tsv_data["Difficult non"] >= 1)]
    clean = temp_data
    lengths = []
    for i in range(len(temp_data)):
        if len(clean["Target word"].iloc[i].split(' ')) == 1:
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

main()
