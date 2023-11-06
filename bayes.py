from dataclasses import dataclass
import re
import math
import datetime
import pandas
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from tqdm import tqdm


WINDOW_SIZE = 50

@dataclass
class Word:
    word: str
    total_spam: float = 0
    total_ham: float = 0
    runs: int = 0
    
    @property
    def average_spam(self):
        return self.total_spam / self.runs

    @property
    def average_ham(self):
        return self.total_ham / self.runs

    def insert_propability(self, spam: float, ham: float):
        self.total_ham += ham
        self.total_spam += spam
        self.runs += 1

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

    def preprocess_word(self, text: str) -> str:
        """
        Preprocesses the input text by removing non-alphanumeric characters and converting it to lowercase.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: A list of preprocessed words in the text.

        """

        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        text = str(text)

        #tokens = word_tokenize(text)
        #stopword_list = stopwords.words("english")
        #tokens = [token for token in tokens if token not in stopword_list]
        #tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]

        #text = " ".join(tokens)

        if text == "":
            raise ValueError("Text is empty after preprocessing.")

        return text

    def preprocess_list(self, text: list) -> list:
        preprocessed_text = []
        print("Preprocessing text...")
        for word in tqdm(text):
            try:
                item = list(word)
                item[0] = str(item[0])
                item[0] = self.preprocess_word(item[0])
                preprocessed_text.append(item)
            except ValueError:
                continue
        return preprocessed_text

    def train(self):
        """
        Trains the Naive Bayes classifier. This method counts the number of words in spam and ham emails
        and calculates the prior probabilities for spam and ham.

        """
        data = self.preprocess_list(self.training_data)
        for word, _, ad in data:
            if ad:
                self.total_spam += 1
                self.spam_word_counts[word] += 1
            else:
                self.total_ham += 1
                self.ham_word_counts[word] += 1

        self.prior_spam = self.total_spam / len(data)
        self.prior_ham = self.total_ham / len(data)

    def visualize_words(self):
        # Generate word cloud for spam words
        spam_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(self.spam_word_counts)

        # Generate word cloud for ham words
        ham_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(self.ham_word_counts)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Display the spam word cloud
        ax1.imshow(spam_wordcloud, interpolation="bilinear")
        ax1.set_title("Spam Word Cloud")
        ax1.axis("off")

        # Display the ham word cloud
        ax2.imshow(ham_wordcloud, interpolation="bilinear")
        ax2.set_title("Ham Word Cloud")
        ax2.axis("off")

        # Show the plot
        plt.tight_layout()
        plt.show()

    def visualize_data_summary(self):
        # Create a figure for the pie chart
        plt.figure(figsize=(6, 6))

        # Create a pie chart for Spam vs. Ham Distribution
        plt.pie(
            [self.total_spam, self.total_ham],
            labels=["Spam", "Ham"],
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Spam vs. Ham Distribution")

        # Show the plot
        plt.show()

    def _classify(self, words: str) -> bool:
        """
        Classifies a text as spam or ham.

        Args:
            text (str): The text to classify.

        Returns:
            list: A list containing the log probability of the text being spam and the log probability of the text being ham.

        """

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

        return [log_prob_spam, log_prob_ham]

    def test(self, testing_data: list, window_size=WINDOW_SIZE) -> None:
        """
        Tests a list of texts and classifies them as spam or ham using a sliding window.

        Args:
            testing_data (list): A list of texts to classify. Each element should contain word, start.
            window_size (int): The size of the sliding window.

        """

        processed_training_data = self.preprocess_list(testing_data)
        words = [Word(a[0]) for a in processed_training_data]

        for index in range(len(processed_training_data[:-window_size + 1])):
            window = [a[0] for a in processed_training_data[index : index + window_size]]
            spam, ham = self._classify(window)

            for word in words[index : index + window_size]:
                word.insert_propability(spam, ham)
        
        if len(processed_training_data) < window_size:
            window = [a[0] for a in processed_training_data]
            spam, ham = self._classify(window)

            for word in words:
                word.insert_propability(spam, ham)

        plt.figure(figsize=(10, 6))

        spam_score = []
        ham_score = []
        timestamps = []

        for index, word in enumerate(words):
            timestamps.append(processed_training_data[index][1])
            spam_score.append(word.average_spam)
            ham_score.append(word.average_ham)

        # Create a plot for the spam and ham scores

        plt.plot(timestamps, spam_score, label="Spam")
        plt.plot(timestamps, ham_score, label="Ham")

        plt.xlabel("Time")
        plt.ylabel("Score")

        plt.legend()

        plt.show()
