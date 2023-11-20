

from typing import Callable
import pandas as pd
from bayes import NaiveBayesClassifier
from preprocessors import DEFAULT_CHUNK_WORDS, preprocess_words
import argparse
from preprocessors import DEFUALT_PREPROCESSORS, DEFAULT_CHUNK_WORDS

def train(input_file:str, output_file:str, chunk_words:int, preprocessors:list[Callable[[str], str]]):

    model = NaiveBayesClassifier()

    training_data = pd.read_csv(input_file, encoding="UTF-8", converters= {'word': lambda x: str(x)})

    clean_data = preprocess_words(training_data, chunk_words, preprocessors)

    model.train(training_data=clean_data)

    model.save(output_file)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Train a model to predict whether a video is an ad or not.")

    argparse.add_argument("-i", "--input-file", type=str, help="The input file to train the model on.")

    argparse.add_argument("-o", "--output-file", type=str, help="The output file to save the model to.", default="model.bayes")

    argparse.add_argument("--chunk-words", type=int, help="The number of words to chunk together.", default=DEFAULT_CHUNK_WORDS)
    argparse.add_argument("--stopwords", type=bool, help="Preprocess with stopwords and lemmatization", default=True)
    argparse.add_argument("--substituion", type=bool, help="Preprocess with substitution.", default=True)
    
    args = argparse.parse_args()

    preprocessors = []

    if args.stopwords:
        preprocessors.append(DEFUALT_PREPROCESSORS[1])
    
    if args.substituion:
        preprocessors.append(DEFUALT_PREPROCESSORS[0])

    print(args)

    train(args.input_file, args.output_file, args.chunk_words, preprocessors)