import pandas as pd
from bayes import (
    DEFAULT_HAM_THRESHOLD,
    NaiveBayesClassifier,
    DEFAULT_WINDOW_SIZE,
    evaluate_classification,
)
from transcribers.audio_transcriber import get_transcription, get_video_id
from preprocessors import DEFUALT_PREPROCESSORS, DEFAULT_CHUNK_WORDS, preprocess_words
import argparse

from transcribers.youtube_transcription_fetcher import fetch_transcript


def main(
    video: str,
    model_data: str,
    chunk_words: int,
    preprocessors: list[str],
    verbose: bool,
    window_size: int,
    ham_threshold: float,
):
    model = NaiveBayesClassifier.load_model(model_data)

    video_id = get_video_id(video)
    data = get_transcription(video_id)
    clean_data = preprocess_words(data, chunk_words, preprocessors)
    classification = model.classify_text(clean_data)

    evaluate_classification(
        classification, verbose=verbose, ham_threshold=ham_threshold
    )


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(
        description="Predict whether a video is an ad or not."
    )

    argparse.add_argument(
        "--video",
        type=str,
        help="The video to predict. Can either be a YouTube video ID or a URL.",
        required=True,
    )
    argparse.add_argument(
        "--model-data",
        type=str,
        help="The model data to use for prediction. Should be preprocessed data.",
        required=True,
    )
    argparse.add_argument(
        "--chunk-words",
        type=int,
        help="The number of words to chunk together.",
        default=DEFAULT_CHUNK_WORDS,
    )
    argparse.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output."
    )
    argparse.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Preprocess with stopwords and lemmatization",
    )
    argparse.add_argument(
        "--no-substitution", action="store_true", help="Preprocess with substitution."
    )
    argparse.add_argument(
        "-w",
        "--window-size",
        help="Window size used to classify text",
        default=DEFAULT_WINDOW_SIZE,
    )
    argparse.add_argument(
        "-t",
        "--ham-threshold",
        help="Ham threshold used to classify text",
        default=DEFAULT_HAM_THRESHOLD,
        type=float,
    )

    args = argparse.parse_args()
    preprocessors = []

    if not args.no_substitution:
        preprocessors.append(DEFUALT_PREPROCESSORS[0])

    if not args.no_stopwords:
        preprocessors.append(DEFUALT_PREPROCESSORS[1])

    print(preprocessors)

    main(
        args.video,
        args.model_data,
        args.chunk_words,
        preprocessors,
        args.verbose,
        args.window_size,
        args.ham_threshold,
    )
