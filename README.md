This repository contains feature extraction code for the paper "Profile-based Authorship Analysis."

The dataset is provided here: https://s3.amazonaws.com/jonathandunn/Legislative_Texts.zip

The Vectorizers in the 'data' folder were trained on speeches from the US House and US Senate, Canadian House, and European Parliament, along with misc. political speeches (all in data set).

This produces X, y feature vectors with or without part-of-speech tags. The "ITFIDF" file produces TF-IDF transforms while the "RAW" file produces frequency counts.