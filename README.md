# Ibiza Chatbot using Deep Learning

Welcome to Ibiza Chatbot, an interactive chatbot powered by deep learning techniques! This chatbot is designed to engage users in conversations and provide responses based on predefined intents.

## Project Overview

The Ibiza Chatbot project involves the development of a chatbot using deep learning techniques, specifically a neural network model built with TensorFlow/Keras. The chatbot is trained on a dataset of intents, where each intent represents a category of user queries or messages. Using this dataset, the chatbot learns to classify incoming messages and generates appropriate responses based on the detected intent.

## Features

1. **Chat Interface:** Users can interact with the chatbot through a simple text input interface.
2. **Intent Classification:** The chatbot classifies user messages into predefined intents using a trained deep learning model.
3. **Response Generation:** Based on the detected intent, the chatbot generates contextually relevant responses to engage users in conversation.

## Data Preparation and Model Training

1. **Data Cleaning:** The project preprocesses the dataset of intents by tokenizing and lemmatizing the words in each message.
2. **Model Architecture:** A deep neural network model is constructed using TensorFlow/Keras, comprising multiple dense layers with dropout regularization.
3. **Training Process:** The model is trained on the preprocessed dataset using the Stochastic Gradient Descent (SGD) optimizer and categorical cross-entropy loss function.

## Used Libraries
`nltk`, `numpy`, `tensorflow`, `streamlit`

## Conclusion

The Ibiza Chatbot project demonstrates the application of deep learning techniques for building an interactive conversational agent. By training a neural network model on a dataset of intents, the chatbot can effectively classify user messages and provide contextually relevant responses. This project serves as a foundation for creating more advanced chatbots and natural language processing applications.

The chatbot's responses can be further improved and expanded by enriching the dataset of intents with more examples and refining the model architecture.
