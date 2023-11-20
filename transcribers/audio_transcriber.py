import yt_dlp
import whisper_timestamped as whisper
import pandas as pd
import requests
import os
import re
from typing import Optional
from dataclasses import dataclass
from pandas import DataFrame
from transcribers.youtube_transcription_fetcher import fetch_transcript


@dataclass(frozen=True)
class Segment:
    start: float
    end: float


def get_video_id(video_id: str) -> str:
    """
    Ensure we got a video-id.

    :param video_id: Either a video-id or an url to a YouTube video.
    :return: Returns the video-id of the YouTube video.
    """
    pattern = r"^[-_a-zA-Z0-9]{11}$"
    match = re.search(pattern, video_id)
    if match:
        return video_id

    pattern = r"[?&]v=([-_a-zA-Z0-9]{11})"
    match = re.search(pattern, video_id)
    if match:
        return match.group(1)
    else:
        raise ValueError("Neither a YouTube video-url or a video-id was supplied.")


def transcribe_video(video_id: str) -> pd.DataFrame:
    """
    Downloads an mp3 of a youtube video, then captions and lables it.

    :param video_id: The id of a YouTube video (the 'v' GET param), or an URL
    :return: Returns a DataFrame with the columns 'word', 'start', and 'ad'
    """
    # Transcribe the video:
    video_id = get_video_id(video_id)
    download_segment(video_id, "tmp")
    df = transcribe_segment("tmp.mp3", False)

    segments = get_ad_segments(video_id)
    if segments:
        segment_i = 0
        segment = segments[segment_i]

        for df_i, row in df.iterrows():
            # Before an ad
            if row["start"] < segment.start:
                continue

            # After an ad
            if row["start"] > segment.end:
                segment_i += 1
                if segment_i >= len(segments):
                    break  # No more ads
                segment = segments[segment_i]
                continue

            # During an ad
            df.at[df_i, "ad"] = True

    return df


def transcribe_ads(video_id: str) -> pd.DataFrame:
    """
    Transcribes only ad segments of a YouTube video if the video has any.

    :param video_id: The id of a YouTube video (the 'v' GET param), or an URL
    :return: Returns a DataFrame with the columns 'word', 'start', and 'ad' for ad segments.
    """
    video_id = get_video_id(video_id)
    ad_segments = get_ad_segments(video_id)

    ad_transcription_df = pd.DataFrame(columns=["word", "start", "ad"])
    for idx, segment in enumerate(ad_segments):
        segment_filename = f"tmp_ad_{idx}"

        download_segment(video_id, segment_filename, segment)
        segment_transcription_df = transcribe_segment(segment_filename + ".mp3", True)

        segment_transcription_df["start"] += segment.start
        segment_transcription_df.dropna(axis=1, how="all", inplace=True)

        if not segment_transcription_df.empty:
            ad_transcription_df = pd.concat(
                [ad_transcription_df, segment_transcription_df], ignore_index=True
            )

    return ad_transcription_df


def get_ad_segments(video_id: str) -> list[Segment]:
    """
    Get a list of segments that represent where in the video an ad occured.
    :param video_id: The id of a YouTube video.
    :return: Returns a list of segemnts in the video that contains an ad.
    """

    r = requests.get(f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}&category=sponsor")
    if r.status_code != 200:
        # Not in the database -> no ads.
        return []

    segments = [Segment(*s["segment"]) for s in r.json()]
    return segments


def download_segment(
    video_id: str, segment_filename: str, segment: Optional[Segment] = None
):
    """
    Saves a specific Youtube video segment.

    :param video_id: The id of a YouTube video (the 'v' GET param), or an URL
    :param segment_filename: The filename to save the downloaded segment.
    :param segment: The start and stop of the segment to of the video to download. Defaults to the entire video.
    """
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": segment_filename,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
    }

    if segment is not None:
        ydl_opts["postprocessor_args"] = (
            [
                "-ss",
                str(segment.start),
                "-to",
                str(segment.end),
            ],
        )

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"http://www.youtube.com/watch?v={video_id}"])


def transcribe_segment(
    segment_filename: str, default_ad_value: bool = False, delete_file: bool = True
) -> pd.DataFrame:
    """
    Transcribes an audio file.

    :param delete_file: Whether to delete the audio file after transcribing it.
    :param segment_filename: The filename of the segment to transcribe.
    :param default_ad_value: The value the 'ad' colum of the DataFrame defaults to.
    :return: Returns a DataFrame with the columns 'word', 'start', and 'ad'.
    """
    audio = whisper.load_audio(segment_filename)
    if delete_file:
        os.remove(segment_filename)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")

    words = []
    starts = []
    ads = []
    for o in result["segments"]:
        for word in o["words"]:
            words.append(word["text"])
            starts.append(word["start"])
            ads.append(default_ad_value)

    return pd.DataFrame({"word": words, "start": starts, "ad": ads})



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
    print(video_id)
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