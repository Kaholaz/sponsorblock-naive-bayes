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
HAM_THRESHOLD = 0.8


@dataclass
class Word:
    word: str
    total_spam_probability: float = 0
    runs: int = 0

    @property
    def average_spam(self):
        return self.total_spam_probability / self.runs

    def insert_propability(self, spam_probability: float):
        self.total_spam_probability += spam_probability
        self.runs += 1


class NaiveBayesClassifier:
    def __init__(self, training_data: list = []):
        """
        Initializes a Naive Bayes classifier with the provided training data.

        Args:
            training_data (list): A list of tuples, each containing a text and its label ("spam" or "ham").

        """

        self.training_data = training_data
        self.spam_word_counts = defaultdict(float)
        self.ham_word_counts = defaultdict(float)
        self.total_spam = 0
        self.total_ham = 0
        self.prior_spam = 0
        self.prior_ham = 0

    def preprocess_word(self, word: str) -> str:
        """
        Preprocesses the input text by removing non-alphanumeric characters and converting it to lowercase.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: A list of preprocessed words in the text.

        """

        clean_word = re.sub(r"[^a-zA-Z\s]", "", word).lower()
        clean_word = str(clean_word)

        # tokens = word_tokenize(text)
        # stopword_list = stopwords.words("english")
        # tokens = [token for token in tokens if token not in stopword_list]
        # tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]

        # text = " ".join(tokens)

        if clean_word == "":
            raise ValueError("Text is empty after preprocessing.")

        return clean_word

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

    def train(self) -> None:
        """
        Trains the Naive Bayes classifier. This method counts the number of words in spam and ham emails
        and calculates the prior probabilities for spam and ham.

        """
        data = self.preprocess_list(self.training_data)

        # Count the number of occurrences of each word in spam and ham emails
        for word, _, ad in data:
            if ad:
                self.total_spam += 1
                self.spam_word_counts[word] += 1
            else:
                self.total_ham += 1
                self.ham_word_counts[word] += 1


        # Calculate prior probabilities (P(spam) and P(ham))
        self.prior_spam = self.total_spam / (self.total_spam + self.total_ham)
        self.prior_ham = self.total_ham / (self.total_spam + self.total_ham)

        # Normalize the word counts, to prevent bias towards longer texts
        self.spam_word_counts = defaultdict(
            lambda: 0,
            {k: v * (1 / self.prior_spam) for k, v in self.spam_word_counts.items()},
        )
        self.ham_word_counts = defaultdict(
            lambda: 0, {k: v * (1 / self.prior_ham) for k, v in self.ham_word_counts.items()}
        )


    def visualize_words(self) -> None:
        # Generate word cloud for spam words
        spam_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(self.spam_word_counts)

        # Generate word cloud for ham words
        ham_wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(self.ham_word_counts)

        # Create subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    def visualize_data_summary(self) -> None:
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

        plt.show()

    def _classify(self, words: list[str]) -> float:
        """
        Classifies a text as spam or ham.

        Args:
            text (str): The text to classify.

        Returns:
            list: A list containing the log probability of the text being spam and the log probability of the text being ham.

        """

        # Find "log-likelihood ratio" of the text being spam 
        # https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Document_classification

        log_spam_likelihood = math.log(self.prior_spam / self.prior_ham)

        for word in words:
            log_spam_likelihood += math.log(
                (self.spam_word_counts[word] + 1) / (self.ham_word_counts[word] + 1)
            )

        # Find P(spam | text) using the log probability
        spam_probability = math.exp(log_spam_likelihood) / (1 + math.exp(log_spam_likelihood))

        return spam_probability

    def test(self, testing_data: list, window_size=WINDOW_SIZE, ham_threshold=HAM_THRESHOLD) -> None:
        """
        Tests a list of texts and classifies them as spam or ham using a sliding window.

        Args:
            testing_data (list): A list of texts to classify. Each element should contain word, start.
            window_size (int): The size of the sliding window.

        """

        processed_training_data = self.preprocess_list(testing_data)
        words = [Word(a[0]) for a in processed_training_data]

        for index in range(len(processed_training_data[: -window_size + 1])):
            window = [
                a[0] for a in processed_training_data[index : index + window_size]
            ]
            spam = self._classify(window)

            for word in words[index : index + window_size]:
                word.insert_propability(spam)

        if len(processed_training_data) < window_size:
            window = [a[0] for a in processed_training_data]
            spam = self._classify(window)

            for word in words:
                word.insert_propability(spam)

        plt.figure(figsize=(10, 6))

        spam_score = []
        timestamps = []

        for index, word in enumerate(words):
            time = datetime.timedelta(seconds=processed_training_data[index][1])
            timestamps.append(processed_training_data[index][1])

            if word.average_spam > ham_threshold:
                print(f"Spam: {time} - {word.word}")

            spam_score.append(word.average_spam)

        # Create a plot for the spam and ham scores

        plt.plot(timestamps, spam_score, label="Spam")

        plt.xlabel("Time")
        plt.ylabel("Score")

        plt.title("Spam Score")

        #Format the x-axis as time

        plt.legend()

        plt.show()
