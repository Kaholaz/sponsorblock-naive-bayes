from pathlib import Path
from .bayes.bayes import (
    NaiveBayesClassifier,
    evaluate_classification,
)
from .bayes.preprocessors import (
    preprocess_words,
)
from typing import Callable
import pandas as pd


def predict(
    video: pd.DataFrame,
    model_data: str,
    chunk_words: int,
    preprocessors: list[str],
    verbose: bool,
    window_size: int,
    ham_threshold: float,
):
    model = NaiveBayesClassifier.load_model(model_data)

    clean_data = preprocess_words(video, chunk_words, preprocessors)
    classification = model.classify_text(clean_data, window_size)

    evaluate_classification(
        classification, verbose=verbose, ham_threshold=ham_threshold
    )


def train(
    input_file: str,
    output_file: str,
    chunk_words: int,
    preprocessors: list[Callable[[str], str]],
):
    training_data = pd.read_csv(
        input_file, encoding="UTF-8", converters={"word": lambda x: str(x)}
    )
    clean_data = preprocess_words(training_data, chunk_words, preprocessors)

    model = NaiveBayesClassifier.train(training_data=clean_data)
    model.save(output_file)


def transcribe(data: pd.DataFrame, save_path: str):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
