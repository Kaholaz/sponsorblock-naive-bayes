from config import ROOT_DIR
from typing import Tuple
import pandas as pd
import fasttext
import csv
import os

modelDir = ROOT_DIR + "/model/"
if not os.path.exists(modelDir):
    os.makedirs(modelDir)

dataset_dir = ROOT_DIR + "/transcriptions/"


class FastTextClassifier:

    def __init__(self):
        self.model = None

    def process_dataset(self, data_path=dataset_dir + "youtube_manual_transcriptions.json",
                        save_path=modelDir + "training_data.txt") -> None:
        """
        Processes the json dataset file into a text file format that is compatible with fasttext

        :param data_path: Path to json file
        :param save_path: Path to save the processed data
        :return:
        """
        with open(data_path, "r") as f:
            temp = pd.read_json(f, lines=True)

            temp = temp[temp["transcript"].str.len() > 0].explode("transcript")

            training_data = pd.DataFrame(list(temp["transcript"])).drop(columns=["start", "duration"])

            training_data["ad"] = training_data["ad"].apply(lambda x: "__label__" + str(x))

            training_data[["ad", "text"]].to_csv(save_path, sep=" ", quotechar="", quoting=csv.QUOTE_NONE,
                                                 escapechar=" ",
                                                 index=False, header=False)

    def train(self, training_data_path=modelDir + "training_data.txt", epochs=5) -> None:
        """
        Trains the model with the processed data

        :param training_data_path: The path of the processed training data
        :param epochs: Number of epochs to train model
        :return:
        """
        self.model = fasttext.train_supervised(training_data_path, epoch=epochs)

    def save_model(self, file_name=modelDir + "fasttext_model") -> None:
        """
        Saves the trained model to a file

        :param file_name: Path to save the model
        :return:
        """
        if self.model:
            self.model.save_model(file_name)
        else:
            print("There is no model to save")

    def load_model(self, path=modelDir + "fasttext_model") -> None:
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

# if __name__ == "__main__":
#    classifier = FastTextClassifier()

#    classifier.process_dataset()

#    classifier.train()

#    classifier.save_model()

#    prediction = classifier.predict("well todays sponsor doesnt think so if you want to learn and earn with coding while simultaneously living a stress free checkout the code boot camp")

#    print(prediction)