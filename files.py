import pandas as pd
from pandas import DataFrame
from enum import Enum
import os
from transcribers.transcribe import transcribe_video, get_video_id
from transcribers.youtube_transcription_fetcher import fetch_transcript


class FileType(Enum):
    TRAINING = "training"
    TESTING = "testing"
    ROOT = ""


DEFAULT_TRANSCRIPTION_PATH = "transcriptions/"


class TranscriptionFileHandler:
    def __init__(self, path=DEFAULT_TRANSCRIPTION_PATH) -> None:
        self.path = path

    def load_data(self, file_type: FileType) -> DataFrame:
        files = os.listdir(f"{self.path}{file_type.value}")

        frames = []
        for file in files:
            with open(f"{self.path}{file_type.value}/{file}", encoding="UTF-8") as f:
                data = pd.read_csv(f)
            frames.append(data)

        return pd.concat(frames)

    def transcribe_and_save_videos(self, videos: [str], file_type: FileType) -> None:
        if not os.path.exists(self.path + file_type.value):
            os.makedirs(self.path + file_type.value)

        for index, video in enumerate(videos):
            video_id = get_video_id(video)
            transcription = transcribe_video(video_id)
            with open(
                    f"{self.path}{file_type.value}/{index}.csv", "w", encoding="UTF-8"
            ) as f:
                transcription.to_csv(f)

    @staticmethod
    def get_transcription(video: str) -> DataFrame:
        video_id = get_video_id(video)
        try:
            transcription = fetch_transcript(video_id)
        except Exception as e:
            print("An error occurred while fetching the transcript: ", e)
            print("\nDownloading and transcribing the video with WhisperX.\n")
            transcription = transcribe_video(video_id)

        return transcription

    def check_resouce_exists(self) -> bool:
        return os.path.exists(self.path)
