import re
import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, training_data: list):
        """
        Initializes a Naive Bayes classifier with the provided training data.

        Args:
            training_data (list): A list of tuples, each containing a text and its label ("spam" or "ham").

        """
        self.training_data = training_data
        self.spam_word_counts = defaultdict(int)
        self.ham_word_counts = defaultdict(int)
        self.total_spam = 0
        self.total_ham = 0
        self.prior_spam = 0
        self.prior_ham = 0

    def preprocess_text(self, text: str) -> list:
        """
        Preprocesses the input text by removing non-alphanumeric characters and converting it to lowercase.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: A list of preprocessed words in the text.

        """
        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        return text.split()

    def train(self):
        """
        Trains the Naive Bayes classifier. This method counts the number of words in spam and ham emails
        and calculates the prior probabilities for spam and ham.

        """
        for text, label in self.training_data:
            words = self.preprocess_text(text)
            if label == "spam":
                self.total_spam += 1
                for word in words:
                    self.spam_word_counts[word] += 1
            else:
                self.total_ham += 1
                for word in words:
                    self.ham_word_counts[word] += 1

        self.prior_spam = self.total_spam / len(self.training_data)
        self.prior_ham = self.total_ham / len(self.training_data)

    def classify(self, text: str) -> bool:
        """
        Classifies a text as spam or ham.

        Args:
            text (str): The text to classify.

        Returns:
            bool: True if spam, False if ham.

        """
        words = self.preprocess_text(text)
        log_prob_spam = math.log(self.prior_spam)
        log_prob_ham = math.log(self.prior_ham)

        # Add log probabilities of each word
        # https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Document_classification
        for word in words:
            log_prob_spam += math.log(
                (self.spam_word_counts[word] + 1)
                / (self.total_spam + len(self.spam_word_counts))
            )
            log_prob_ham += math.log(
                (self.ham_word_counts[word] + 1)
                / (self.total_ham + len(self.ham_word_counts))
            )

        return log_prob_spam > log_prob_ham

    def test(self, testing_data: list) -> None:
        """
        Tests a list of texts and classifies them as spam or ham.

        Args:
            testing_data (list): A list of texts to classify.

        """
        for text in testing_data:
            prediction = self.classify(text)
            label = "spam" if prediction else "ham"
            print(f"Text: '{text}' - Predicted: {label}")
