import pandas as pd
from enum import Enum
import os
from transcribers.transcribe import transcribe_video, get_video_id
from dataclasses import dataclass


class FileType(Enum):
    TRAINING = "training"
    TESTING = "testing"
    ROOT = ""


@dataclass
class AdTaggedWord:
    word: str
    start: float
    ad: bool


DEFAULT_TRANSCRIPTION_PATH = "transcriptions/"


class TranscriptionFileHandler:
    def __init__(self, path=DEFAULT_TRANSCRIPTION_PATH) -> None:
        self.path = path

    def load_data(self, file_type: FileType) -> list[AdTaggedWord]:
        training_data = []

        files = os.listdir(f"{self.path}{file_type.value}")

        for file in files:
            with open(f"{self.path}{file_type.value}/{file}", encoding="UTF-8") as f:
                data = pd.read_csv(f)
            training_data.extend(
                AdTaggedWord(*z) for z in zip(data["word"], data["start"], data["ad"])
            )

        return training_data

    def transcribe_and_save_videos(self, videos: [str], file_type: FileType) -> None:
        if not os.path.exists(self.path + file_type.value):
            os.makedirs(self.path + file_type.value)

        for index, video in enumerate(videos):
            video_id = get_video_id(video)
            transcription = transcribe_video(video_id)
            with open(
                f"{self.path}{file_type.value}/{index}.csv", "w", encoding="UTF-8"
            ) as f:
                transcription.to_csv(f, index=False)

    def check_resouce_exists(self) -> bool:
        return os.path.exists(self.path)
