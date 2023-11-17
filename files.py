import pandas as pd
from pandas import DataFrame
from enum import Enum
import os
from transcribers.transcribe import transcribe_video, get_video_id
from transcribers.youtube_transcription_fetcher import fetch_transcript




def load_multiple_files(dir: str) -> DataFrame:
    files = os.listdir(dir)

    frames = []
    for file_path in files:
        with open(file_path, encoding="UTF-8") as f:
            data = pd.read_csv(f)
        frames.append(data)

    return frames

def transcribe_and_save_videos(videos: [str], output_file_name: str) -> None:
    frames = []
    for video in videos:
        video_id = get_video_id(video)
        transcription = transcribe_video(video_id)
        frames.append(transcription)

    with open(output_file_name, "w", encoding="UTF-8"
    ) as f:
        transcription = pd.concat(frames)
        transcription.to_csv(f)

def get_transcription(video: str) -> DataFrame:
    video_id = get_video_id(video)
    try:
        transcription = fetch_transcript(video_id)
    except Exception as e:
        print("An error occurred while fetching the transcript: ", e)
        print("\nDownloading and transcribing the video with WhisperX.\n")
        transcription = transcribe_video(video_id)

    return transcription

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
