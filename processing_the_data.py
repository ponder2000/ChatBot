"""Importing the datasets"""
import nltk         #Natural Language Tool Kit
from nltk.stem.lancaster import LancasterStemmer    #for finding the root word of a given word
import numpy as np
import json
import pickle

# reading the hotcoded dataset from a json fromat
with open('custom_dataset_for_special_perpose_chatbot.json') as file:
    data = json.load(file)

# Extracting data from json file to the following variables
# we need the class/tag of the words, a list of unique word in the sentance
words = []      # list of all root words (basically the words that are known to the bot)
labels = []     # collection of all type of main tags
docs_x = []     # docs_x will have queries in token form
docs_y = []     # docs_y will have the respective tags of the queries

# tokanizing the words in each queries using nltk.word_tokenizer
for intent in data['intents']:
    for query in intent['queries']:
        tokanized_wrds = nltk.word_tokenize(query)
        words.extend(tokanized_wrds)
        docs_x.append(tokanized_wrds)
        docs_y.append(intent['tag'])

    # filling the labels with all available tags
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

"""
Now we have the list of words, tags(labels)
Now it's time to process the data that we extracted
"""
# here our words list is still not rooted, so we need to stem it
stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w!="?"]
words = sorted(list(set(words)))
labels = sorted(labels)

""" the model will not be able to train with the string hence we need to convert it into numerical vector.
    creating a word_vector that will have length of words list,
    if the word is present in query and known to the machine then it will have 1 otherwise 0 for unknown words
"""
training_questions = []
training_tag = []
temp_tag_vector = [0 for _ in range(len(labels))]

for i, doc in enumerate(docs_x):
    word_vector = []
    stemmed_wrds = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        # converting into a word vector
        if w in stemmed_wrds:
            word_vector.append(1)
        else:
            word_vector.append(0)

    tag_vector = temp_tag_vector[:]
    tag_vector[labels.index(docs_y[i])] = 1

    training_questions.append(word_vector)
    training_tag.append(tag_vector)

# now we have the data in numerical form so for better result we will conver it into numpy arrays
training_questions = np.array(training_questions)
training_tag = np.array(training_tag)

with open("saved_data.pickle", "wb") as f:
        pickle.dump((words, labels, training_questions, training_tag), f)

print("Data processing completed")