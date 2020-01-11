import nltk         #Natural Language Tool Kit
from nltk.stem.lancaster import LancasterStemmer    #for finding the root word of a given word
import numpy as np
import time
import json
import tflearn
import tensorflow
import pickle

# importing the processed data
with open("saved_data.pickle", "rb") as f:
    words, labels, training_questions, training_tag = pickle.load(f)

"""
Up till now our data is preprocessed now its time to feed it into the NN
"""
# creating a neural Network
tensorflow.reset_default_graph()

network = tflearn.input_data(shape=[None, len(words)])
network = tflearn.fully_connected(network, len(labels) + 6)
network = tflearn.fully_connected(network, len(labels) + 6)
network = tflearn.fully_connected(network, len(labels), activation="softmax")
network = tflearn.regression(network)

nn_model = tflearn.DNN(network)

# training the model and then saving it for future use
print("Training the model")
start = time.time()
nn_model.fit(training_questions, training_tag, n_epoch=1000, batch_size=(len(labels) + 6), show_metric=True)
end = time.time()
print("Saving the model")
nn_model.save("nn_model.tflearn")
print("Model saved")
print("Time required for training {}".format(end-start))
