import spacy
import sys
import numpy as np
import pandas as pd
from collections import Counter
import textacy

def main():
    print("PART A: Linguistic analysis using spaCy")
    print(f"Version of Spacy used for development 3.2.4 Current verion used: {spacy.__version__}")
    print(f"Version of Python used for development 3.8.13 Current version used: {sys.version}")

    # Question 1. Tokenization
    print("\nQuestion 1. Tokenization")
    nlp = spacy.load("en_core_web_sm")

    with open ("data/preprocessed/train/sentences.txt", encoding = "utf8") as text:
        data = text.readlines()
        string = ""
        for i in range(len(data)):
            string += str(data[i]).replace("\n", " ").replace("\\", "")
    doc = nlp(string)

    # Number of tokens
    count = 0
    for token in doc:
        count+=1
    print(f"Number of tokens {count}")

    # Number of types
    count = 0
    unique = []
    
    for words in doc:
        unique.append(words.text)
            
    unique = np.unique(unique)    
    print(f"The number of types {len(unique)}")
    
    # Number of words
    skip = ["PUNCT"]
    word_list = []
    for token in doc:
        if not token.is_punct:
            word_list.append(token.text)
    print(f"Number of words: {len(word_list)}")

    # Average of words per sentence
    wordcount = len(word_list)
    sentcount = len(list(doc.sents))
    print(f"The average words per sentence is {wordcount/sentcount}")
    
    word_lengths = []
    for i in word_list:
        word_lengths.append(len(i))
    print(f"The average word length is {np.mean(word_lengths)} +/- {np.std(word_lengths)}")


    # Question 2. 
    print("\nQuestion 2. Word Classes")

    tokens = [token.tag_ for token in doc]
    tokens = np.array(tokens)
    tokens = np.unique(tokens)
    token_count = dict.fromkeys(tokens, 0)
    for token in doc:
        token_count[token.tag_] += 1
    
    my_keys = sorted(token_count, key=token_count.get, reverse=True)[:10]
    uberdict = dict.fromkeys(my_keys, 0)

    for i in my_keys:
        # get universal POS applicable to finegrained POS
        uniPOS = [token.pos_
            for token in doc if token.tag_ == i]
        uniPOS = list(set(uniPOS))
        # get all words
        all_words = [token.text
                for token in doc if token.tag_ ==i]
         # get count of words for finegrained POS token
        fineg_count = len(all_words)
        # create dict with all words existing for this finegrained POS token
        words = np.array(all_words)
        words = np.unique(words)
        word_dict = dict.fromkeys(words, 0) 
        # count number of occurences of word 
        for tok in all_words:
            word_dict[tok] += 1
            # get most common and least common words
            keys_freqtokens = sorted(word_dict, key=word_dict.get, reverse=True)[:3]
            keys_unfreqtokens = sorted(word_dict, key=word_dict.get)[:1]
            uberdict[i] = [uniPOS, fineg_count, keys_freqtokens, keys_unfreqtokens]

    total_tags = 0
    for i in uberdict.values():
        total_tags += i[1]
    
        for i in uberdict.keys():
            uberdict[i].append(uberdict[i][1]/total_tags)
    df = pd.DataFrame.from_dict(uberdict,orient="index")
    print(df)


    # Question 3.
    print("\nQuestion 3. N-Grams")


    print("\n Question 4. Lemmatization")


    print("\n Question 5. Named Entity Recognition")

# Create sentence object and an array for entity information
    doc_sent = doc.sents
    array = np.zeros(len(list(doc.sents))).tolist()

# Loop over sentences save entities in array
    counter = 0
    first_five = []
    for sent in doc.sents:
        first_five.append(sent)
        temp_entity = []
        sent_text = nlp(str(sent))

        for ent in sent_text.ents:
            temp_entity.append([ent.text, ent.label_])
        array[counter] = temp_entity
        counter += 1
    
    # Get entities and entity labels back from array
    named_entities = 0
    entity_labels = []
    
    for i in range(len(array)):
        temp_array = np.array(array[i]).T
        if temp_array.size != 0:
            named_entities += len(temp_array[0]) + 1
            entity_labels.append(temp_array[1].tolist())
                                        
    print(f"Number of named entities: {named_entities}")    
    print(f"Number of different entity labels: {len(set(np.hstack(entity_labels).tolist()))}")
    
    
    # Analyse first five sentences
    for i in range(0,5):
        print(f"Sentence {i+1}:")
        print(first_five[i])
        if np.array(array[i]).T.size != 0:
            print(f"Entities: {np.array(array[i]).T[0]}")
            print(f"Labels: {np.array(array[i]).T[1]}")
                                    
        print()
    



main()
