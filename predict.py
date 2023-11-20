import pandas as pd
from bayes import NaiveBayesClassifier
from transcribers.audio_transcriber import get_transcription, get_video_id, transcribe_video
from preprocessors import DEFUALT_PREPROCESSORS, DEFAULT_CHUNK_WORDS
import argparse

from transcribers.youtube_transcription_fetcher import fetch_transcript

def main(video: str, model_data: str, chunk_words: int, preprocessors: list[str]):
    model = NaiveBayesClassifier()

    video_id = get_video_id(video)
    model.load(model_data)

    data = fetch_transcript(video_id)

    classification = model.classify_text(data)
    
    model.evaluate_classification(classification)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Predict whether a video is an ad or not.")

    argparse.add_argument("--video", type=str, help="The video to predict. Can either be a YouTube video ID or a URL.", required=True)

    argparse.add_argument("--model-data", type=str, help="The model data to use for prediction. Should be preprocessed data.",required=True)

    argparse.add_argument("--chunk-words", type=int, help="The number of words to chunk together.", default=DEFAULT_CHUNK_WORDS)
    argparse.add_argument("--stopwords", type=bool, help="Preprocess with stopwords and lemmatization", default=True)
    argparse.add_argument("--substitution", type=bool, help="Preprocess with substitution.", default=True)
    
    args = argparse.parse_args()

    preprocessors = []

    if args.stopwords:
        preprocessors.append(DEFUALT_PREPROCESSORS[1])
    
    if args.substitution:
        preprocessors.append(DEFUALT_PREPROCESSORS[0])

    args = argparse.parse_args()

    preprocessors = []

    if args.stopwords:
        preprocessors.append(DEFUALT_PREPROCESSORS[1])
    
    if args.substitution:
        preprocessors.append(DEFUALT_PREPROCESSORS[0])

    main(args.video, args.model_data, args.chunk_words, preprocessors)
