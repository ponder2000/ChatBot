import nltk         #Natural Language Tool Kit
from nltk.stem.lancaster import LancasterStemmer    #for finding the root word of a given word
import numpy as np
import time
import json
import tflearn
import tensorflow
import pickle
import random

# reading the hotcoded dataset from a json fromat
with open('custom_dataset_for_special_perpose_chatbot.json') as file:
    data = json.load(file)

# reading the crime data for getting the info about a crime
with open('crimes_info.json') as file:
    crime_data = json.load(file)

# reding the help file
with open('help_data.txt','r') as f:
    help_data = f.read()

# importing the processed data
with open("saved_data.pickle", "rb") as f:
    words, labels, training_questions, training_tag = pickle.load(f)

# creating a neural Network
tensorflow.reset_default_graph()

network = tflearn.input_data(shape=[None, len(words)])
network = tflearn.fully_connected(network, len(labels) + 6)
network = tflearn.fully_connected(network, len(labels) + 6)
network = tflearn.fully_connected(network, len(labels), activation="softmax")
network = tflearn.regression(network)

nn_model = tflearn.DNN(network)

# loading the pre_trained model
nn_model.load("nn_model.tflearn")

# creating stem object
stemmer = LancasterStemmer()

# converting the query into a word vector
def query_to_word_vector(query, words):
    wrds_vector = [0 for _ in range(len(words))]

    modified_query = nltk.word_tokenize(query)
    modified_query = [stemmer.stem(w.lower()) for w in modified_query]

    for q_wrd in modified_query:
        for i,wrd in enumerate(words):
            if wrd == q_wrd:
                wrds_vector[i] = 1
            
    return np.array(wrds_vector)

# creating a awareness function about crime
def crime_information():
    global crime_data
    tag = input("Bot: Please specify the category of crime, I didn't get you properly\nYou: ")
    tag = tag.lower()
    found_query = False
    for crime in crime_data['crimes']:
        if crime['crime_name'] in tag:
            found_query = True
            print("BOT: {}".format(crime['defination']))
            break
    if not found_query:
        print("BOT: Sorry but it seems like you entered an invalid crime!")
    return

# creating a FIR/complaint registration function
def FIR_registraion():
    pass

def help_function():
    global help_data
    print("BOT: {}".format(help_data))
    return

def chat():
    print("Welcome to the AI chat bot!")
    while True:
        query = input("You: ")
        if query.lower() == "quite":
            break
        pred_tag = nn_model.predict([query_to_word_vector(query, words)])[0]
        pred_tag_indx = np.argmax(pred_tag)

        if pred_tag[pred_tag_indx] > 0.6:
            # bot is confident about the query
            tag = labels[pred_tag_indx]
            if tag == "information":
                crime_information()     # this function will print the info about the crime asked
            
            elif tag == "help":
                help_function()

            elif tag == "register_crime":
                # define a function for FIR registration
                print("--------WILL BE NAVIGATED TO FIR CREATION FUNCTION---------")
            
            else:
                for intent in data['intents']:
                    if intent['tag'] == tag:
                        probable_response = intent['responses']
                print("BOT: {}".format(random.choice(probable_response)))
        else:
            # bot is confused
            print("BOT: Sorry but I didn't get you!")
            with open("unresolved_query.txt","a") as f:
                f.write(query + "\n")

chat()

