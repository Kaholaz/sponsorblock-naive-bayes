from dataclasses import dataclass, field
import re
import math
import datetime
from collections import defaultdict
from typing import Optional, Callable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

import numpy as np
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


_stopword_list = stopwords.words("english")
_word_lemmatizer = WordNetLemmatizer()


def stopword_preprocessor(word: str):
    return " ".join(
        _word_lemmatizer.lemmatize(word)
        for w in word.lower().split()
        if w not in (_stopword_list)
    )


DEFAULT_WINDOW_SIZE = 100
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

        print("Chunking words...")
        if chunk_words > 1:
            text["chunked_word"] = text["word"]
            text["shifted_word"] = text["word"]
            for _ in range(1, chunk_words):
                text["shifted_word"] = text["shifted_word"].shift(-1)
                text["chunked_word"] = text["chunked_word"] + " " + text["shifted_word"]

            text["word"] = text["chunked_word"]
            text.drop(columns=["chunked_word", "shifted_word"], inplace=True)
            text = text.iloc[: -chunk_words + 1]

        print("Preprocessing text...")
        clean_text = text
        for preprocessor in preprocessors:
            clean_text["word"] = clean_text["word"].progress_apply(preprocessor)

        clean_text = clean_text.loc[clean_text["word"] != ""]
        clean_text.reset_index(drop=True, inplace=True)

        return clean_text

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

        spam_words = list(training_data[training_data["ad"] == True]["word"])
        self.total_spam = len(spam_words)

        ham_words = list(training_data[training_data["ad"] == False]["word"])
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
        self, testing_data: DataFrame, window_size=DEFAULT_WINDOW_SIZE, alpha=DEFAULT_ALPHA
    ) -> DataFrame:
        """
        Tests a list of texts and classifies them as spam or ham using a sliding window.
        :arg testing_data: A list of texts to classify_window. Each element should contain word, start.
        :arg window_size: The size of the sliding window.
        :arg ham_threshold: The threshold for classifying a word as spam.
        :return: Returns a list of classified words.
        """

        testing_data.insert(3, "total_spam", [0.0] * len(testing_data))
        testing_data.insert(4, "runs", [0] * len(testing_data))

        words = list(testing_data["word"])
        if len(words) < window_size:
            window_size = len(words)
        # Use a sliding window to classify_window the words
        for index in trange(len(testing_data.index) - window_size + 1):
            spam = self.classify_window(words[index : index + window_size], alpha)

            # Insert the spam probability for each word in the window
            testing_data.loc[index : index + window_size - 1, "total_spam"] += spam
            testing_data.loc[index : index + window_size - 1, "runs"] += 1

        return testing_data

    @staticmethod
    def evaluate_classification(words: DataFrame, ham_threshold=DEFAULT_HAM_THRESHOLD):
        timestamps = []
        spam_score = []
        real_spam_score = []
        failed_predictions = 0
        false_negatives = 0
        false_positives = 0

        print(f"Spam words ({ham_threshold} threshold):")
        for index, word in words.iterrows():
            timestamps.append(word["start"])
            average_spam = word["total_spam"] / word["runs"]
            spam_score.append(average_spam)
            real_spam_score.append(float(word.ad))

            is_ad = word["ad"]

            false_negative = is_ad and average_spam < ham_threshold
            false_positive = not is_ad and average_spam > ham_threshold

            if false_negative:
                false_negatives += 1
                failed_predictions += 1

            if false_positive:
                false_positives += 1
                failed_predictions += 1

            if average_spam > ham_threshold:
                timestamp = datetime.timedelta(seconds=word["start"])
                max_width = 20  # Adjust this value to your desired column width
                value = word["word"]
                print(
                    f"Spam: {str(timestamp).ljust(max_width)}Word: {value.ljust(max_width)}Ad: {str(is_ad).ljust(max_width)}Average spam: {average_spam}"
                )
        print("\nAccuracy:         ", 1 - (failed_predictions / len(words)))
        print("False positives (%):", false_positives / failed_predictions)
        print("False negatives (%):", false_negatives / failed_predictions)
        print("Total words:       ", len(words))
        print("Total ads:         ", len(words[words["ad"] == True]))
        print("\nParameters:")
        print("Ham threshold:     ", ham_threshold)

        plot_spam_score(timestamps, spam_score, real_spam_score)

    def visualize_word_importance(self, start_n, stop_n) -> None:
        """
        Visualizes the importance of words for spam classification.
        Displays the top N words with their importance scores.
        
        :param top_n: The number of top words to display.
        """

        all_words = set(self.spam_word_counts.keys()) | set(self.ham_word_counts.keys())

        spam_word_array = np.array([self.spam_word_counts[word] for word in all_words])
        ham_word_array = np.array([self.ham_word_counts[word] for word in all_words])

        importance_scores = (spam_word_array +1) / (ham_word_array + 1)

        top_indices = np.argsort(importance_scores)[::-1][start_n:stop_n]

        top_words = np.array(list(all_words))[top_indices]
        top_scores = importance_scores[top_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(top_words, top_scores, color="blue")
        plt.title("Top Words Importance for Spam Classification")
        plt.xlabel("Word")
        plt.ylabel("Importance Score")

        # Show the plot
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


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


def visualize_top_words(model: NaiveBayesClassifier, top_n: int = 10) -> None:

    # Get the top n spam words
    top_spam_words = sorted(
        model.spam_word_counts.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    # Get the top n ham words
    top_ham_words = sorted(
        model.ham_word_counts.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    # Create subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display the top spam words
    ax1.bar(
        [word[0] for word in top_spam_words],
        [word[1] for word in top_spam_words],
        color="red",
    )
    ax1.set_title("Top Spam Words")
    ax1.set_xticklabels([word[0] for word in top_spam_words], rotation=45)
    ax1.set_ylabel("Count")

    # Display the top ham words
    ax2.bar(
        [word[0] for word in top_ham_words],
        [word[1] for word in top_ham_words],
        color="green",
    )
    ax2.set_title("Top Ham Words")
    ax2.set_xticklabels([word[0] for word in top_ham_words], rotation=45)
    ax2.set_ylabel("Count")

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
