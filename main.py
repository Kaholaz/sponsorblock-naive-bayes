from bayes import NaiveBayesClassifier
from transcribe import *


def main():
    # transcribe_data()
    training_data = []
    for index in range(1, 3):
        with open(f"transcriptions/{index}.csv", "r") as f:
            data = pd.read_csv(f)
        training_data.extend(list(zip(data["word"], data["start"], data["ad"])))

    testing_data = []
    with open(f"transcriptions/0.csv", "r") as f:
        data = pd.read_csv(f)
    testing_data = list(zip(data["word"], data["start"], data["ad"]))
    bayes = NaiveBayesClassifier(training_data=testing_data)

    bayes.train()
    bayes.test(testing_data=testing_data)


def transcribe_data():
    videos = [
        "https://www.youtube.com/watch?v=8rofPMY54Ko&t=207s",
        "https://www.youtube.com/watch?v=mXBzBFxe00o",
        "https://www.youtube.com/watch?v=jCuEBVbmPcA",
        "https://www.youtube.com/watch?v=r4NL2e6jJ04",
    ]

    for index, video in enumerate(videos):
        video_id = get_video_id(video)
        transcription = transcribe(video_id)
        with open(f"transcriptions/{index}.csv", "w") as f:
            transcription.to_csv(f, index=False)


if __name__ == "__main__":
    main()
