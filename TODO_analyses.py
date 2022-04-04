# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy
nlp = spacy.load("en_core_web_sm")

#"data/preprocessed/train/sentectes.txt"




with open ("data/preprocessed/train/sentences.txt", encoding = "utf8") as text:
    
    data = text.readlines()
    string = ''.join([str(item) for item in data])

    doc = nlp(string)


nouns =  [chunk.text for chunk in doc.noun_chunks]
print(f"Noun = {len(nouns)}")

#print("Noun", [chunk.text for chunk in doc.noun_chunks])
#print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

