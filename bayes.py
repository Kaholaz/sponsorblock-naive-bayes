from dataclasses import dataclass, field
from functools import cached_property
import re
import math
import datetime
from collections import defaultdict
from typing import Optional, Callable
from files import AdTaggedWord
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from tqdm import tqdm, trange


def substitution_preprocessor(word: str) -> str:
    """
    Preprocesses the input text by removing non-alphanumeric characters and converting it to lowercase.

    :arg text: The text to preprocess.
    :return: The preprocessed text.

    """
    clean_word = str(word)
    clean_word = re.sub(r"[^a-zA-Z\s]", "", clean_word).lower()

    return clean_word


def stopword_preprocessor(word: str) -> str:
    """
    Remove all words in the stopword list provided by nltk.corpus.stopwords

    :param word: A single word
    :return: Returns "" if the word is filtered out, else return the word.
    """
    tokens = word_tokenize(word)
    stopword_list = stopwords.words("english")
    tokens = [token for token in tokens if token not in stopword_list]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    clean_word = " ".join(tokens)

    return clean_word


WINDOW_SIZE = 50
HAM_THRESHOLD = 0.5
DEFAULT_WORD_CHUNKING = 2
ALPHA = 1
DEFAULT_PREPROCESSORD = [substitution_preprocessor]


@dataclass
class Word(AdTaggedWord):
    total_spam_probability: float = 0
    runs: int = 0

    @cached_property
    def average_spam(self):
        return self.total_spam_probability / self.runs

    def insert_propability(self, spam_probability: float):
        self.total_spam_probability += spam_probability
        self.runs += 1


@dataclass
class NaiveBayesClassifier:
    training_data: list[AdTaggedWord] = field(default_factory=list)
    spam_word_counts: defaultdict[float] = field(
        default_factory=lambda: defaultdict(float)
    )
    ham_word_counts: defaultdict[float] = field(
        default_factory=lambda: defaultdict(float)
    )
    total_spam: int = 0
    total_ham: int = 0
    prior_spam: float = 0
    prior_ham: float = 0

    @staticmethod
    def preprocess_words(
        text: list[AdTaggedWord],
        chunk_words: int = DEFAULT_WORD_CHUNKING,
        preprocessors: Optional[list[Callable[[str], str]]] = None,
    ) -> list[Word]:
        """
        Preprocesses by running a list of preprocessors on each word in a list of words.

        :param text: A list of unprocessesd words. Defaults to substitution_preprocessor.
        :return: A list of processed words.
        """
        if preprocessors is None:
            preprocessors = [substitution_preprocessor]

        preprocessed_text = []
        print("Preprocessing text...")
        for i in trange(len(text) - chunk_words + 1):
            word_chunk = text[i : i + chunk_words]
            clean_chunk = []
            for word in word_chunk:
                clean_word = word.word
                for preprocessor in preprocessors:
                    clean_word = preprocessor(clean_word)

                if clean_word != "":
                    clean_chunk.append(clean_word)

            if len(clean_chunk):
                start_word = text[i]
                word = Word(
                    word=" ".join(clean_chunk), start=start_word.start, ad=start_word.ad
                )
                preprocessed_text.append(word)

        return preprocessed_text

    def train(self) -> None:
        """
        Trains the Naive Bayes classifier. This method counts the number of words in spam and ham emails
        and calculates the prior probabilities for spam and ham.
        """

        clean_data = self.preprocess_words(self.training_data)

        # Count the number of occurrences of each word in spam and ham emails
        print("Training...")
        for word in tqdm(clean_data):
            if word.ad:
                self.total_spam += 1
                self.spam_word_counts[word.word] += 1
            else:
                self.total_ham += 1
                self.ham_word_counts[word.word] += 1

        # Calculate prior probabilities (P(spam) and P(ham))
        self.prior_spam = self.total_spam / (self.total_spam + self.total_ham)
        self.prior_ham = self.total_ham / (self.total_spam + self.total_ham)

        # Normalize the word counts, to prevent bias towards longer texts
        self.spam_word_counts = defaultdict(
            lambda: 0,
            {k: v * (1 / self.prior_spam) for k, v in self.spam_word_counts.items()},
        )
        self.ham_word_counts = defaultdict(
            lambda: 0,
            {k: v * (1 / self.prior_ham) for k, v in self.ham_word_counts.items()},
        )

    def classify_window(self, words: list[Word], alpha: float = ALPHA) -> float:
        """
        Classifies a list of words as spam or ham.

        :arg text: The text to classify.
        :arg alpha: The value to add to
        :return: Returns a float between 0 and 1, indicating the probability of the text being spam.
        """

        # Find "log-likelihood ratio" of the text being spam
        # https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Document_classification

        log_spam_likelihood = math.log(self.prior_spam / self.prior_ham)

        for word in words:
            log_spam_likelihood += math.log(
                (self.spam_word_counts[word.word] + alpha)
                / (self.ham_word_counts[word.word] + alpha)
            )

        # Find P(spam | text) using the log probability
        spam_probability = math.exp(log_spam_likelihood) / (
            1 + math.exp(log_spam_likelihood)
        )

        return spam_probability

    def classify_text(
        self, testing_data: list[AdTaggedWord], window_size=WINDOW_SIZE
    ) -> list[Word]:
        """
        Tests a list of texts and classifies them as spam or ham using a sliding window.
        :arg testing_data: A list of texts to classify_window. Each element should contain word, start.
        :arg window_size: The size of the sliding window.
        :arg ham_threshold: The threshold for classifying a word as spam.
        :return: Returns a list of classified words.
        """

        clean_data = self.preprocess_words(testing_data)

        # Make sure the window size is not larger than the data
        if len(clean_data) < window_size:
            window_size = len(clean_data)

        # Use a sliding window to classify_window the words
        for index in range(len(clean_data[: -window_size + 1])):
            window = clean_data[index : index + window_size]
            spam = self.classify_window(window)

            # Insert the spam probability for each word in the window
            for window_index in range(len(window)):
                word = clean_data[window_index + index]
                word.insert_propability(spam)

        return clean_data

    @staticmethod
    def evaluate_classification(words: list[Word], ham_threshold=HAM_THRESHOLD):
        timestamps = []
        spam_score = []
        real_spam_score = []
        failed_predictions = 0

        print(f"Spam words ({ham_threshold} threshold):")
        for index, word in enumerate(words):
            timestamps.append(word.start)
            spam_score.append(word.average_spam)
            real_spam_score.append(float(word.ad))

            false_negative = words[index].ad and word.average_spam < ham_threshold
            false_positive = not words[index].ad and word.average_spam > ham_threshold
            if false_negative or false_positive:
                failed_predictions += 1

            if word.average_spam > ham_threshold:
                timestamp = datetime.timedelta(seconds=word.start)
                max_width = 20  # Adjust this value to your desired column width
                print(
                    f"Spam: {str(timestamp).ljust(max_width)}Word: {word.word.ljust(max_width)}Ad: {str(word.ad).ljust(max_width)}Average spam: {word.average_spam}"
                )
        print("\nAccuracy:", 1 - (failed_predictions / len(words)))
        plot_spam_score(timestamps, spam_score, real_spam_score)


def visualize_words(model: NaiveBayesClassifier) -> None:
    # Generate word cloud for spam words
    spam_wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(model.spam_word_counts)

    # Generate word cloud for ham words
    ham_wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(model.ham_word_counts)

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


def visualize_data_summary(model: NaiveBayesClassifier) -> None:
    # Create a figure for the pie chart
    plt.figure(figsize=(6, 6))

    # Create a pie chart for Spam vs. Ham Distribution
    plt.pie(
        [model.total_spam, model.total_ham],
        labels=["Spam", "Ham"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Spam vs. Ham Distribution")

    plt.show()


def plot_spam_score(
    timestamps: list[int], spam_score: list[float], real_spam_score: list[float]
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, spam_score, label="Spam prediction")
    plt.plot(timestamps, real_spam_score, label="Real spam")

    plt.title("Spam Score")
    plt.xlabel("Time")
    plt.ylabel("Score")

    plt.legend()

    plt.show()
