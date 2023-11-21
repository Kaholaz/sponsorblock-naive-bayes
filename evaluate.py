from pandas import DataFrame

from bayes import (
    NaiveBayesClassifier,
    evaluate_classification,
    visualize_words,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_HAM_THRESHOLD,
    visualize_data_summary,
)
from transcribers.audio_transcriber import get_transcription, transcribe_segment
from preprocessors import DEFUALT_PREPROCESSORS, DEFAULT_CHUNK_WORDS, preprocess_words
import argparse


def main(
    _data: DataFrame,
    _model_data: str,
    _chunk_words: int,
    _preprocessors: list[callable],
    _window_size: int,
    _ham_threshold: float,
):
    model = NaiveBayesClassifier.load_model(_model_data)

    print("Creating word cloud.")
    visualize_words(model)
    visualize_data_summary(model)

    clean_data = preprocess_words(_data, _chunk_words, _preprocessors)
    classification = model.classify_text(clean_data, window_size=_window_size)

    evaluate_classification(classification, ham_threshold=_ham_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the model on content. Either a YouTube video, local "
        "audio file or a transcription csv file."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-v",
        "--video",
        type=str,
        help="The video to evaluate. Can either be a YouTube video ID or a URL.",
    )
    group.add_argument(
        "-a",
        "--audio-path",
        type=str,
        help="For evaluating a local audio file. Has to be the path of the file.",
    )
    group.add_argument(
        "-d",
        "--transcription-path",
        type=str,
        help="For evaluating the model on a transcription file. Has to be the path of the file.",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        help="The path of the model file to evaluate on. Should be preprocessed data.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--chunk-words",
        type=int,
        help="N-gram/number of words to chunk together, defaults to "
        + str(DEFAULT_CHUNK_WORDS)
        + ".",
        default=DEFAULT_CHUNK_WORDS,
        required=False,
    )
    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        help="The size of the rolling window used for classification, defaults to "
        + str(DEFAULT_WINDOW_SIZE)
        + ".",
        default=DEFAULT_WINDOW_SIZE,
        required=False,
    )
    parser.add_argument(
        "-ht",
        "--ham-threshold",
        type=float,
        help="The threshold to be exceeded for classifying something as spam, defaults to "
        + str(DEFAULT_HAM_THRESHOLD)
        + ".",
        default=DEFAULT_HAM_THRESHOLD,
        required=False,
    )
    parser.add_argument(
        "--stopwords",
        type=bool,
        help="Preprocess with stopwords and lemmatization, defaults to True.",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--substitution",
        type=bool,
        help="Preprocess with substitution, defaults to True.",
        default=True,
        required=False,
    )

    args = parser.parse_args()

    if args.video:
        data = get_transcription(args.video)
    elif args.audio_path:
        data = transcribe_segment(args.audio_path, delete_file=False)
    elif args.dataset_path:
        data = DataFrame.from_csv(args.transcription_path)
    else:
        raise Exception(
            "Missing args. Must provide something to transcribe, either --video, --audio-path or --transcription-path"
        )

    preprocessors = []

    if args.stopwords:
        preprocessors.append(DEFUALT_PREPROCESSORS[1])

    if args.substitution:
        preprocessors.append(DEFUALT_PREPROCESSORS[0])

    main(
        data,
        args.model_path,
        args.chunk_words,
        preprocessors,
        args.window_size,
        args.ham_threshold,
    )
