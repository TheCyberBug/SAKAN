import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from transformers import GPT2Tokenizer
from tqdm import tqdm

class DataPreprocessor:
    """
    A class for preprocessing text data for sentiment analysis.
    
    This class provides methods to load, clean, lemmatize, and tokenize text data.
    """
    def __init__(self):
        # NLTK data init
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self, train_path, test_path):
        """
        Load training and test data from CSV files.

        Args:
            train_path (str): Path to the training CSV file.
            test_path (str): Path to the test CSV file.

        Returns:
            tuple: A tuple containing train and test DataFrames.
        """
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test

    def transform_label(self, label):
        """
        Transform sentiment labels to binary format.

        Args:
            label (str): The sentiment label ('positive' or 'negative').

        Returns:
            int: 1 for positive sentiment, 0 for negative sentiment.
        """
        return 1 if label == 'positive' else 0

    def preprocess_text(self, text):
        """
        Preprocess the input text by lowercasing, removing HTML tags,
        special characters, and stop words.

        Args:
            text (str): input text.

        Returns:
            str: The preprocessed text.
        """
        text = text.lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        words = word_tokenize(text)
        sent = [word for word in words if word not in self.stop_words]
        return ' '.join(sent)

    def lemmatize_text(self, text):
        """
        Lemmatize the words in the input text.

        Args:
            text (str): input text.

        Returns:
            str: lemmatized text.
        """
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def tokenize_text(self, text, max_length=256):
        """
        Tokenize the input text using the GPT-2 tokenizer.

        Args:
            text (str): input text.
            max_length (int, optional): Maximum length of the tokenized sequence. Defaults to 256.

        Returns:
            list: A list of token IDs.
        """
        return self.tokenizer.encode(
            text=text,
            add_special_tokens=False,
            truncation=True,
            add_prefix_space=True,
            padding='max_length',
            max_length=max_length
        )

    def process_data(self, data):
        """
        Process the input DataFrame by applying all preprocessing steps.

        Args:
            data (pd.DataFrame): The input DataFrame containing 'sentiment' and 'review' columns.

        Returns:
            pd.DataFrame: The processed DataFrame with additional columns for cleaned, 
                          lemmatized, and tokenized text, as well as binary labels.
        """
        tqdm.pandas()
        data['label'] = data['sentiment'].progress_apply(self.transform_label)
        data['clean'] = data.review.progress_apply(self.preprocess_text)
        data['lemma'] = data['clean'].progress_apply(self.lemmatize_text)
        data['tokenized'] = data.lemma.progress_apply(self.tokenize_text)
        return data
