from bayes import NaiveBayesClassifier
from files import TranscriptionFileHandler, FileType
from transcribe import *
import nltk

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
#download_nltk_data()


def main():
    #transcribe_ads_data()
    training_videos = [
        "https://www.youtube.com/watch?v=mXBzBFxe00o",
        "https://www.youtube.com/watch?v=jCuEBVbmPcA",
        "https://www.youtube.com/watch?v=r4NL2e6jJ04",
    ]

    testing_videos = [
        "https://www.youtube.com/watch?v=8rofPMY54Ko&t=207s",
    ]

    files = TranscriptionFileHandler()

    if not files.check_resouce_exists():
        files.transcribe_and_save_videos(training_videos, FileType.TRAINING)
        files.transcribe_and_save_videos(testing_videos, FileType.TESTING)

    training_data = files.load_data(FileType.TRAINING)
    testing_data = files.load_data(FileType.TESTING)

    bayes = NaiveBayesClassifier(training_data=training_data)

    bayes.train()

    #bayes.visualize_words()
    #bayes.visualize_data_summary()

    bayes.test(testing_data=testing_data)



def transcribe_ads_data():
    videos = [
            "https://www.youtube.com/watch?v=mXBzBFxe00o",
        ]
    for index, video in enumerate(videos):
        video_id = get_video_id(video)
        transcription = transcribe_ads(video_id)
        if not transcription.empty:
            with open(f"transcriptions/{video_id}_ads.csv", "w") as f:
                transcription.to_csv(f, index=False)
        else:
            print(f"No ads to transcribe for video {video_id}")


if __name__ == "__main__":
    main()
