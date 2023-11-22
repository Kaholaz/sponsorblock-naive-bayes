import argparse
from bayes.bayes import (
    DEFAULT_HAM_THRESHOLD,
    NaiveBayesClassifier,
    DEFAULT_WINDOW_SIZE,
    evaluate_classification,
)
from bayes.preprocessors import (
    DEFUALT_PREPROCESSORS,
    DEFAULT_CHUNK_WORDS,
    preprocess_words,
)
import pandas as pd
from config import ROOT_DIR
from pathlib import Path
from transcribers.audio_transcriber import get_transcription, transcribe_segment
from cli import predict, train, transcribe

parser = argparse.ArgumentParser(
    description="Transcribe YouTube videos, train a model, or predict the ad content of a video."
)
preprocess_parser = argparse.ArgumentParser(add_help=False)
preprocess_parser.add_argument(
    "--chunk-words",
    type=int,
    help="The number of words to chunk together.",
    default=DEFAULT_CHUNK_WORDS,
)
preprocess_parser.add_argument(
    "--no-stopwords",
    action="store_true",
    help="Preprocess with stopwords and lemmatization",
)
preprocess_parser.add_argument(
    "--no-substitution", action="store_true", help="Preprocess with substitution."
)

SAVE_PATH = ROOT_DIR + "/transcriptions/transcription.csv"
sub_parsers = parser.add_subparsers(title="action", dest="action")
predict_parser = sub_parsers.add_parser(
    name="predict",
    description="Predict whether a video is an ad or not.",
    parents=[preprocess_parser],
)
train_parser = sub_parsers.add_parser(
    name="train",
    description="Train a model to predict whether a video is an ad or not.",
    parents=[preprocess_parser],
)
transcribe_parser = sub_parsers.add_parser(
    name="transcribe",
    description="Transcribes a youtube video or audio file, and saves transcription to a path.",
)

# Predict
predict_input_group = predict_parser.add_mutually_exclusive_group(required=True)
predict_input_group.add_argument(
    "-y",
    "--video",
    type=str,
    help="For transcribing YouTube video. Can either be a YouTube video ID or "
    "a URL.",
)
predict_input_group.add_argument(
    "-a",
    "--audio-path",
    type=str,
    help="For transcribing a local audio file. Has to be absolute path "
    "of the audio file.",
)
predict_input_group.add_argument(
    "-d",
    "--transcription-path",
    type=str,
    help="For evaluating the model on a transcription file. Has to be the path of the file.",
)

predict_parser.add_argument(
    "--model-data",
    type=str,
    help="The model data to use for prediction. Should be preprocessed data.",
    required=True,
)
predict_parser.add_argument(
    "-v", "--verbose", action="store_true", help="Print verbose output."
)
predict_parser.add_argument(
    "-w",
    "--window-size",
    help="Window size used to classify text",
    default=DEFAULT_WINDOW_SIZE,
)
predict_parser.add_argument(
    "-t",
    "--ham-threshold",
    help="Ham threshold used to classify text",
    default=DEFAULT_HAM_THRESHOLD,
    type=float,
)

# Train
train_parser.add_argument(
    "-i",
    "--input-file",
    type=str,
    help="The input file to train the model on.",
    required=True,
)
train_parser.add_argument(
    "-o",
    "--output-file",
    type=str,
    help="The output file to save the model to.",
    default="model.bayes",
)

# Transcribe
transcribe_input_group = transcribe_parser.add_mutually_exclusive_group(required=True)
transcribe_input_group.add_argument(
    "-v",
    "--video",
    type=str,
    help="For transcribing YouTube video. Can either be a YouTube video ID or "
    "a URL.",
)
transcribe_input_group.add_argument(
    "-a",
    "--audio-path",
    type=str,
    help="For transcribing a local audio file. Has to be absolute path "
    "of the audio file.",
)
transcribe_parser.add_argument(
    "-s",
    "--save-path",
    type=str,
    help="Optional full file path to save to, including filetype. If "
    "ommitted, defaults the save path to /transcriptions folder in "
    "project root.",
    required=False,
    default=SAVE_PATH,
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.action in ("train", "predict"):
        preprocessors = []
        if not args.no_substitution:
            preprocessors.append(DEFUALT_PREPROCESSORS[0])
        if not args.no_stopwords:
            preprocessors.append(DEFUALT_PREPROCESSORS[1])

    if args.action in ("transcribe", "predict") and args.video:
        if args.video:
            data = get_transcription(args.video)
        elif args.audio_path:
            data = transcribe_segment(args.audio_path, delete_file=False)
        elif args.action == "predict" and args.transcription_path:
            data = pd.DataFrame.from_csv(args.transcription_path)

    if args.action == "transcribe":
        transcribe(data, args.save_path)
    elif args.action == "predict":
        predict(
            data,
            args.model_data,
            args.chunk_words,
            preprocessors,
            args.verbose,
            args.window_size,
            args.ham_threshold,
        )
    elif args.action == "train":
        train(args.input_file, args.output_file, args.chunk_words, preprocessors)
