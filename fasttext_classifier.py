from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from pandas import DataFrame
from files import TranscriptionFileHandler

from config import ROOT_DIR
from typing import Tuple
import pandas as pd
import fasttext
import re
import csv
import os

modelDir = ROOT_DIR + "/model/"
if not os.path.exists(modelDir):
    os.makedirs(modelDir)

dataset_dir = ROOT_DIR + "/transcriptions/"

THRESHOLD = 0.98

STOPWORDS = stopwords.words("english")

def preprocessor(data: DataFrame) -> DataFrame:
    """
    Preprocesses the data, lemmatizes and removes unnecessary symbols and stopwords.

    :param data: DataFrame containing text to preprocess
    :return: Returns the preprocessed DataFrame
    """

    data = data.apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]*", "", x.lower()))
    data = data.apply(lambda x: " ".join(
        [WordNetLemmatizer().lemmatize(word) for word in x.split() if word not in STOPWORDS]))
    return data[data.str.strip().astype(bool)]


class FastTextClassifier:

    def __init__(self):
        self.model = None

    @staticmethod
    def process_dataset(data_path: str = dataset_dir + "youtube_manual_transcriptions.csv",
                        save_path: str = modelDir + "training_data.txt") -> None:
        """
        Processes a dataframe with the columns word, start, ad into a text file format that is compatible with fasttext

        :param data_path: Path to json file
        :param save_path: Path to save the processed data
        :return:
        """
        with open(data_path, "r", encoding="utf-8") as f:
            temp = pd.read_csv(f)

            temp = temp.dropna()

            training_data = {
                "label": [],
                "text": []
            }

            print("Loaded dataset, starting to process...")
            reg1 = re.compile(r"[^a-zA-Z0-9\s]*")

            current_is_ad = True
            current_sentence = []
            current_start = 0
            for index, row in temp.iterrows():
                if row["ad"] != current_is_ad or row["start"] < current_start:
                    if current_sentence != "":
                        training_data["text"].append(" ".join(current_sentence))
                        training_data["label"].append("__label__" + str(current_is_ad))
                        current_sentence = []
                    current_is_ad = row["ad"]
                    current_start = row["start"]

                current_sentence.append(reg1.sub("", row["word"]).strip().lower())

            print("Finished merging into sentences, preprocessing now.")
            df_training_data = DataFrame(training_data)

            df_training_data["text"] = preprocessor(df_training_data["text"])

            print("Finished preprocessing, saving to file.")
            df_training_data.to_csv(save_path, sep=" ", quotechar="", quoting=csv.QUOTE_NONE,
                                    escapechar=" ",
                                    index=False, header=False)
            print("Finished saving to file.")

    def train(self, training_data_path: str = modelDir + "training_data.txt", epochs: int = 5) -> None:
        """
        Trains the model with the processed data

        :param training_data_path: The path of the processed training data
        :param epochs: Number of epochs to train model
        :return:
        """
        self.model = fasttext.train_supervised(training_data_path, epoch=epochs)

    def save_model(self, file_name: str = modelDir + "fasttext_model") -> None:
        """
        Saves the trained model to a file

        :param file_name: Path to save the model
        :return:
        """
        if self.model:
            self.model.save_model(file_name)
        else:
            print("There is no model to save")

    def load_model(self, path: str = modelDir + "fasttext_model") -> None:
        """
        Loads a previously saved model from a file

        :param path: Path where the model was saved
        :return:
        """
        self.model = fasttext.load_model(path)

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predicts whether a sentence is an ad or not based on previous training

        :param text: The text to predict
        :return: A tuple that contains the label and the probability
        """
        if self.model:
            return self.model.predict(text)
        else:
            print("Train or load a model first")

    def find_ad_segments(self, data: DataFrame) -> list:
        ad_segments = []
        n = len(data)
        i = 0
        window_size = 10

        data = data[data["word"].str.strip().astype(bool)]

        data["word"] = preprocessor(data["word"])
        #data["word"] = data["word"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]*", "", x.lower()))

        while i < n:
            window = data.iloc[i:min(i + window_size, n-1)]
            window_text = " ".join(window["word"].tolist())
            print(1, window_text)
            prediction, precision = self.predict(window_text)
            if prediction[0] == "__label__True" and precision[0] > THRESHOLD:
                start_time = window.iloc[0]["start"]
                print(precision)
                print(2, window_text)

                d = i
                while i < n and prediction[0] == "__label__True" and precision[0] > THRESHOLD:
                    i += window_size
                    if i < n:
                        window_text = " ".join(data.iloc[d:min(i, n-1)]["word"].tolist())
                        prediction, precision = self.predict(window_text)
                        print(precision)
                        print(3, window_text)

                end_time = data.iloc[min(i - 1, n-1)]["start"]

                if end_time - start_time > 10:
                    ad_segments.append((start_time, end_time))
            else:
                i += window_size

        return ad_segments


if __name__ == "__main__":
    classifier = FastTextClassifier()

    classifier.process_dataset()

    classifier.train(epochs=10)

    classifier.save_model()

    classifier.load_model()

    transcription = TranscriptionFileHandler.get_transcription("https://www.youtube.com/watch?v=GIgc4viI2Vg")

    print(transcription)

    segments = classifier.find_ad_segments(transcription)

    print(segments)
