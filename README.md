# Kolmogorov-Arnold Networks for Sentiment Analysis

## Overview

This project evaluates the suitability of Kolmogorov-Arnold Neural Networks (KANs) for Natural Language Processing tasks, specifically sentiment analysis. We implement a hybrid CNN-KAN architecture and test it on the IMDB movie review dataset.

## Features

- Sentiment analysis using a CNN-KAN hybrid architecture
- Data preprocessing and tokenization using GPT2Tokenizer
- Implementation of KAN layers using FastKAN library
- Experiments with dropout and data augmentation techniques
- Comparative evaluation against baseline CNN and CNN+LSTM models

## Requirements

- Python 3.8+
- PyTorch
- NLTK
- Transformers (Hugging Face)
- pandas
- matplotlib
- tqdm
- FastKAN

## Installation

1. Clone this repository.
2. Install fastKAN as a site package from https://github.com/ZiyaoLi/fast-kan
3. Download the IMDB review dataset into the data folder: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
4. Run data/split.py to split the dataset into training/testing.
5. run src/main.py for model training.

## Archive
The archive directory stores historical versions and some older experiments of the project.
The final solution is entirely implemented under the SAKAN/src folder.

## Model-Architecture History (non-runnable).ipynb
This file stores notable phases of the architectural changes during the experimentation phase.
THE NOTEBOOK IS NOT MADE TO BE EXECUTABLE!!!

It only stores the model configurations.
If you want to test one of them, copy it and place it as a new model class in the src/model.py file.
Then change the selected model from src/sakan.py (`model = CustomCNN(vocab_size, embed_dim).to(device)`)
