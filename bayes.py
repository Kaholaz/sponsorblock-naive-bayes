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
import nltk


WINDOW_SIZE = 50


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

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text by removing non-alphanumeric characters and converting it to lowercase.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: A list of preprocessed words in the text.

        """

        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        text = str(text)

        return text

        tokens = word_tokenize(text)
        stopword_list = stopwords.words("english")
        tokens = [token for token in tokens if token not in stopword_list]
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]

        text = " ".join(tokens)

        if text == "":
            raise ValueError("Text is empty after preprocessing.")

        return text

    def train(self):
        """
        Trains the Naive Bayes classifier. This method counts the number of words in spam and ham emails
        and calculates the prior probabilities for spam and ham.

        """

        for word, _, ad in self.training_data:
            try:
                word_pre_processed = self.preprocess_text(word)
            except ValueError:
                continue
            if ad:
                self.total_spam += 1
                self.spam_word_counts[word_pre_processed] += 1
            else:
                self.total_ham += 1
                self.ham_word_counts[word_pre_processed] += 1

        self.prior_spam = self.total_spam / len(self.training_data)
        self.prior_ham = self.total_ham / len(self.training_data)

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

    def classify(self, text: str) -> bool:
        """
        Classifies a text as spam or ham.

        Args:
            text (str): The text to classify.

        Returns:
            list: A list containing the log probability of the text being spam and the log probability of the text being ham.

        """
        try:
            words = self.preprocess_text(text)
        except ValueError:
            return [0, 0]

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

        word_classification = testing_data
        buffer = ""
        spam_total = 0
        ham_total = 0

        for index, (word, _, _) in enumerate(testing_data):
            buffer += word
            if index >= window_size:
                buffer = buffer[
                    len(word) :
                ]  # remove the word at the beginning of the buffer

                spam, ham = self.classify(buffer)

                # Update the totals
                spam_total += spam
                ham_total += ham

                for i in range(index - window_size, index):
                    if isinstance(word_classification[i], tuple):
                        word_classification[i] = list(word_classification[i])
                        word_classification[i].append([0, 0])
                        word_classification[i].append(0)
                        word_classification[i][2] = [0, 0]

                    # Total spam and ham for each word
                    word_classification[i][2][0] += spam
                    word_classification[i][2][1] += ham

                    # Total number of times the word was classified
                    word_classification[i][4] += 1

                    # Average spam and ham for each word
                    word_classification[i][3][0] = (
                        word_classification[i][2][0] / word_classification[i][4]
                    )

                    word_classification[i][3][1] = (
                        word_classification[i][2][1] / word_classification[i][4]
                    )
        plt.figure(figsize=(10, 6))

        spam_score = []
        ham_score = []
        timestamps = []

        for word in word_classification:
            if isinstance(word, tuple):
                continue
            timestamps.append(word[1])
            spam_score.append(word[3][0])
            ham_score.append(word[3][1])
        
        # Create a plot for the spam and ham scores


        plt.plot(timestamps, spam_score, label="Spam")
        plt.plot(timestamps, ham_score, label="Ham")

        plt.show()
        
        