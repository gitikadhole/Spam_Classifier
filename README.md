SMS Spam Classifier
This repository contains a machine learning model for classifying SMS messages as spam or non-spam.

The SMS Spam Classifier is a project aimed at automatically identifying spam messages in SMS communications.
It utilizes a machine learning algorithm trained on a labeled dataset of SMS messages to predict 
whether a given message is spam or non-spam. The classifier can be helpful in various scenarios, such as filtering unwanted messages, protecting users from scams, 
and improving overall user experience.

Dataset
Dataset was downloaded from Kaggle

Model Training
The model training process involves several steps:
Data preprocessing: Cleaning and transforming the raw SMS messages into a suitable format for training.
Feature extraction: Converting the preprocessed messages into numerical features that can be used by the machine learning algorithm.
Model training: Training a machine learning model (Multinomial Bayes) using the labeled SMS messages.
Model evaluation: Assessing the performance of the trained model using various metrics (e.g., accuracy, precision).
