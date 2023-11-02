import re
import math
import datetime
import pandas
from collections import defaultdict

BUFFER_SIZE = 100


class NaiveBayesClassifier:
    def __init__(self, training_data: list = None):
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

    def load_training_data(self, path: str) -> None:
        """
        Loads a CSV file containing training data.

        Data must be of the following format:
        word, start, ad

        Args:
            path (str): The path to the CSV file.

        Returns:
            list: A list of tuples, each containing a text and its label ("spam" or "ham").

        """

        data = pandas.read_csv(path)
        self.training_data = list(zip(data["word"], data["start"], data["ad"]))

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text by removing non-alphanumeric characters and converting it to lowercase.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: A list of preprocessed words in the text.

        """

        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        return text

    def train(self):
        """
        Trains the Naive Bayes classifier. This method counts the number of words in spam and ham emails
        and calculates the prior probabilities for spam and ham.

        """

        for word, start, ad in self.training_data:
            word_pre_processed = self.preprocess_text(word)
            if ad:
                self.total_spam += 1
                self.spam_word_counts[word_pre_processed] += 1
            else:
                self.total_ham += 1
                self.ham_word_counts[word_pre_processed] += 1

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

    def next_buffer(self, testing_data, buffer_size):
        """
        Tests a list of texts and classifies them as spam or ham.
        #TODO: This should do a sliding window instead of a buffer

        Args:
            testing_data (list): A list of texts to classify. Each element should contain word, start.

        """

        buffer = ""
        buffer_start = testing_data[0][1]
        for index, (word, start, ad) in enumerate(testing_data):
            buffer += word
            if index % buffer_size == 0:
                yield buffer, buffer_start
                buffer_start = start
                buffer = ""

    def test(self, testing_data: list, buffer_size=BUFFER_SIZE) -> None:
        """
        Tests a list of texts and classifies them as spam or ham.

        Args:
            testing_data (list): A list of texts to classify. Each element should contain word, start.

        """

        for buffer, buffer_start in self.next_buffer(testing_data, buffer_size):
            if self.classify(buffer):
                start_datetime = datetime.timedelta(seconds=buffer_start)
                print(f"Spam: {start_datetime}")
