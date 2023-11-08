from dataclasses import dataclass, field
import re
import math
import datetime
from collections import defaultdict
from typing import Optional, Callable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from tqdm import tqdm

tqdm.pandas()


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


DEFAULT_WINDOW_SIZE = 50
DEFAULT_HAM_THRESHOLD = 0.5
DEFAULT_WORD_CHUNKING = 1
DEFAULT_ALPHA = 1
DEFAULT_PREPROCESSORS = [substitution_preprocessor, stopword_preprocessor]


@dataclass
class NaiveBayesClassifier:
    word_chunking: int = DEFAULT_WORD_CHUNKING
    preprocessors: list[Callable[[str], str]] = field(
        default_factory=lambda: DEFAULT_PREPROCESSORS
    )

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
        text: DataFrame,
        chunk_words: int = DEFAULT_WORD_CHUNKING,
        preprocessors: Optional[list[Callable[[str], str]]] = None,
    ) -> DataFrame:
        """
        Preprocesses by running a list of preprocessors on each word in a list of words.

        :param text: A list of unprocessesd words. Defaults to substitution_preprocessor.
        :return: A list of processed words.
        """
        if preprocessors is None:
            preprocessors = DEFAULT_PREPROCESSORS

        print("Preprocessing text...")
        clean_text = text
        for preprocessor in preprocessors:
            clean_text["word"] = clean_text["word"].progress_apply(preprocessor)

        return clean_text[clean_text["word"] != ""]

    def train(self, training_data: DataFrame) -> None:
        """
        Trains the Naive Bayes classifier. This method counts the number of words in spam and ham emails
        and calculates the prior probabilities for spam and ham.
        :param training_data: The data to train the model on.
        """
        # Reset model
        self.spam_word_counts = defaultdict(float)
        self.ham_word_counts = defaultdict(float)
        self.total_spam = 0
        self.total_ham = 0
        self.prior_spam = 0
        self.prior_ham = 0

        clean_text = self.preprocess_words(training_data)

        spam_words = list(clean_text[clean_text["ad"] == True]["word"])
        self.total_spam = len(spam_words)

        ham_words = list(clean_text[clean_text["ad"] == False]["word"])
        self.total_ham = len(ham_words)

        for word in tqdm(spam_words, desc="Counting spam words..."):
            self.spam_word_counts[word] += 1

        for word in tqdm(ham_words, desc="Counting ham words..."):
            self.ham_word_counts[word] += 1

        # Calculate prior probabilities (P(spam) and P(ham))
        self.prior_spam = self.total_spam / (self.total_spam + self.total_ham)
        self.prior_ham = self.total_ham / (self.total_spam + self.total_ham)

        # Normalize the word counts, to get the probability given spam/ham
        self.spam_word_counts = defaultdict(
            lambda: 0,
            {k: v * (1 / self.prior_spam) for k, v in self.spam_word_counts.items()},
        )
        self.ham_word_counts = defaultdict(
            lambda: 0,
            {k: v * (1 / self.prior_ham) for k, v in self.ham_word_counts.items()},
        )

    def classify_window(self, words: list[str], alpha: float = DEFAULT_ALPHA) -> float:
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
                (self.spam_word_counts[word] + alpha)
                / (self.ham_word_counts[word] + alpha)
            )

        # Find P(spam | text) using the log probability
        spam_probability = math.exp(log_spam_likelihood) / (
            1 + math.exp(log_spam_likelihood)
        )

        return spam_probability

    def classify_text(
        self, testing_data: DataFrame , window_size=DEFAULT_WINDOW_SIZE
    ) -> DataFrame:
        """
        Tests a list of texts and classifies them as spam or ham using a sliding window.
        :arg testing_data: A list of texts to classify_window. Each element should contain word, start.
        :arg window_size: The size of the sliding window.
        :arg ham_threshold: The threshold for classifying a word as spam.
        :return: Returns a list of classified words.
        """

        clean_data = self.preprocess_words(testing_data)

        clean_data.insert(3, "total_spam", [0.0] * len(clean_data))
        clean_data.insert(4, "runs", [0] * len(clean_data))

        # Make sure the window size is not larger than the data
        clean_data_len = len(clean_data.index)
        if  clean_data_len < window_size:
            window_size = clean_data_len

        words = list(clean_data["word"])
        # Use a sliding window to classify_window the words
        for index in range(clean_data_len - window_size + 1):
            spam = self.classify_window(words[index : index + window_size])

            # Insert the spam probability for each word in the window
            clean_data.loc[index : index + window_size, "total_spam"] += spam
            clean_data.loc[index : index + window_size, "runs"] += 1

        return clean_data

    @staticmethod
    def evaluate_classification(words: DataFrame, ham_threshold=DEFAULT_HAM_THRESHOLD):
        timestamps = []
        spam_score = []
        real_spam_score = []
        failed_predictions = 0

        print(f"Spam words ({ham_threshold} threshold):")
        for index, word in words.iterrows():
            timestamps.append(word["start"])
            average_spam = word["total_spam"] / word["runs"]
            spam_score.append(average_spam)
            real_spam_score.append(float(word.ad))

            is_ad = word["ad"]
            false_negative = is_ad and average_spam < ham_threshold
            false_positive = not is_ad and average_spam > ham_threshold
            if false_negative or false_positive:
                failed_predictions += 1

            if average_spam > ham_threshold:
                timestamp = datetime.timedelta(seconds=word["start"])
                max_width = 20  # Adjust this value to your desired column width
                value = word["word"]
                print(
                    f"Spam: {str(timestamp).ljust(max_width)}Word: {value.ljust(max_width)}Ad: {str(is_ad).ljust(max_width)}Average spam: {average_spam}"
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
