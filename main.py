from bayes import NaiveBayesClassifier, visualize_words, visualize_data_summary
from files import TranscriptionFileHandler, FileType
from transcribers.transcribe import get_video_id, transcribe_ads
import pandas as pd
import nltk


def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")


# download_nltk_data()


def main():
    # transcribe_ads_data()
    training_videos = [
        "https://www.youtube.com/watch?v=mXBzBFxe00o",
        "https://www.youtube.com/watch?v=jCuEBVbmPcA",
        "https://www.youtube.com/watch?v=r4NL2e6jJ04",
    ]

    testing_videos = [
        "https://www.youtube.com/watch?v=8rofPMY54Ko&t=207s",
    ]

    model = NaiveBayesClassifier()
    files = TranscriptionFileHandler()

    if not files.check_resouce_exists():
        files.transcribe_and_save_videos(training_videos, FileType.TRAINING)
        files.transcribe_and_save_videos(testing_videos, FileType.TESTING)
    
    if not files.check_resouce_exists(FileType.PREPROCESSED):
        training_data = files.load_data(FileType.TRAINING)

        clean_training_data = model.preprocess_words(pd.concat(training_data))

        files.dump_preprocessed_words(clean_training_data)
    else:
        clean_training_data = pd.concat(files.load_data(FileType.PREPROCESSED))

    model.train(training_data=clean_training_data)

    visualize_words(model)
    visualize_data_summary(model)

    testing_data = files.load_data(FileType.TESTING)
    for frame in testing_data:
        clean_data = model.preprocess_words(frame)
        classification = model.classify_text(testing_data=clean_data)
        model.evaluate_classification(classification)

if __name__ == "__main__":
    main()
