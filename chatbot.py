import streamlit as st
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words =[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the class of the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
    return return_list

# Get a response from the bot
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Web App for Chatbot using Streamlit
def display_message(user, message):
    st.write(f"**{user}:** {message}")

def display_response(bot, response):
    st.write(f"**{bot}:** {response}")

st.title('Ibiza Chatbot')

user_message = st.text_input('You:', key='user_input')
button_clicked = st.button('Send', key='send_button')

# Create an empty container for displaying messages
message_container = st.empty()

if button_clicked and user_message:
    display_message('You', user_message)
    bot_ints = predict_class(user_message)
    bot_response = get_response(bot_ints, intents)
    display_response('Ibiza Chatbot', bot_response)

    # Create a new input area for the next message
    new_user_message = st.text_input('You:', key='new_user_input')
    new_button_clicked = st.button('Send', key='send_new_button')

    if new_button_clicked and new_user_message:
        display_message('You', new_user_message)
        new_bot_ints = predict_class(new_user_message)
        new_bot_response = get_response(new_bot_ints, intents)
        display_response('Ibiza Chatbot', new_bot_response)