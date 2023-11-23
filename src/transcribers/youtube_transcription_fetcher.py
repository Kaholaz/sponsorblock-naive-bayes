"""
    This code is used to build a dataset of YouTube video transcripts, using youtube api to
        fetch the transcriptions.
    The code uses the filtered and sorted sponsorTimes.csv file to find the ids of videos that
        have sponsor segments.
    Those ids are used with the youtube api to fetch and assign an "ad" label to each line of the transcript.
    It first tries to find manually user written transcripts, otherwise defaults to automatically generated ones.
    Each time a transcript is fetched, each sentence is compared against the sponsor timestamps to assign
        ad labels, after a transcript has been handled, it is saved to the json dataset file specific to
        the type of transcript (manual or auto).
"""
import json
import os
import re

import pandas as pd
import requests
from pandas import DataFrame
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from config import ROOT_DIR

processedSponsorTimesDir = ROOT_DIR + "/transcribers/sponsor_data/"
processedSponsorTimesPath = processedSponsorTimesDir + "processed_sponsorTimes.csv"
os.makedirs(processedSponsorTimesDir, exist_ok=True)

transcriptionDir = ROOT_DIR

manualTranscriptionPath = transcriptionDir + "/youtube_manual_transcriptions.ndjson"
autoTranscriptionPath = transcriptionDir + "/youtube_auto_transcriptions.ndjson"
noTranscriptPath = transcriptionDir + "/yt_vids_without_transcript.csv"

transcriptLang = ["en-GB", "en-US", "en-CA", "en-AU", "en-NZ", "en"]

videoIDCountToCheck = 50000

try:
    dfNoTranscript = pd.read_csv(noTranscriptPath)
    noTranscriptSet = set(dfNoTranscript["videoID"])
except FileNotFoundError:
    noTranscriptSet = set()
    pd.DataFrame(columns=["videoID"]).to_csv(noTranscriptPath, index=False)


def video_id_exists_in_file(file_path: str, video_id: str) -> bool:
    """
    Checks if a video id exists in a file

    :param file_path: The path to the file to check
    :param video_id: The video id to check for
    :return: True if video id exists, otherwise False
    """
    try:
        with open(file_path, "r") as file:
            return video_id in file.read()
    except FileNotFoundError:
        return False


def build_dataset() -> None:
    """
    Builds a dataset of video transcripts by using video ids from processed sponsorTimes.csv to fetch.
    Creates a dataset for both manually translated transcripts, or if there are none,
        it checks for auto generated ones.

    Everything is saved inside the ../transcriptions/ directory.
    :return:
    """
    df_video_sponsor_data = pd.read_csv(
        processedSponsorTimesPath
    )  # , nrows=videoID_count_to_check)

    grouped_sponsor_data = df_video_sponsor_data.groupby("videoID", sort=False)
    # for video_id, group in groupedSponsorData:
    #    print(video_id, "\n", group, zip(group["startTime"], group["endTime"]))

    reg1 = re.compile(r"\s*[(\[](\s|\w)*[])]")
    reg2 = re.compile(r"([\n\"\-,^\s._])+")
    for video_id, group in grouped_sponsor_data:
        all_sponsor_segments = list(zip(group["startTime"], group["endTime"]))
        video_id = str(video_id)

        if video_id_exists_in_file(
            manualTranscriptionPath, video_id
        ) or video_id_exists_in_file(autoTranscriptionPath, video_id):
            print(f"Transcript for videoID {video_id} already exists, skipping fetch")
            continue

        if video_id in noTranscriptSet:
            print(
                f"Skipping videoID {video_id}, as no transcript existed when checked during a past run"
            )
            continue

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_manually_created_transcript(
                transcriptLang
            ).fetch()
            transcription_path = manualTranscriptionPath
        except NoTranscriptFound:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_generated_transcript(
                    transcriptLang
                ).fetch()
                transcription_path = autoTranscriptionPath
            except NoTranscriptFound:
                print(
                    f"No manually or automatically generated transcript exists for video ID {video_id}"
                )
                noTranscriptSet.add(video_id)
                with open(noTranscriptPath, "a") as csv_file:
                    csv_file.write(video_id + "\n")
                continue
            except Exception as e:
                print(f"An error occurred for video ID {video_id}: {e}")
                continue

        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video ID {video_id}")
            noTranscriptSet.add(video_id)
            with open(noTranscriptPath, "a") as csv_file:
                csv_file.write(video_id + "\n")
            continue
        except Exception as e:
            print(f"An error occurred for video ID {video_id}: {e}")
            continue

        processed_transcript = {"video_id": video_id, "transcript": []}
        for line in transcript:
            clean_text = reg1.sub(" ", line["text"]).strip()
            clean_text = reg2.sub(" ", clean_text).strip()
            if not clean_text:
                continue

            line_start = line["start"]
            line_end = line["start"] + line["duration"]

            is_ad = False
            for sponsor_start, sponsor_end in all_sponsor_segments:
                # print(f"Line: {line} | Sponsor: {sponsor_start} - {sponsor_end}")
                # print(f"{line_start < sponsor_end-0.2} and {line_end > sponsor_start} is ad?")
                if line_start < sponsor_end - 0.2 and line_end > sponsor_start:
                    is_ad = True
                    break

            processed_transcript["transcript"].append(
                {
                    "text": clean_text,
                    "start": line["start"],
                    "duration": line["duration"],
                    "ad": is_ad,
                }
            )

        with open(transcription_path, "a") as json_file:
            json_file.write(json.dumps(processed_transcript) + "\n")
            print(f"Saved transcript for videoID {video_id}")


def dataset_file_to_df(file_path: str) -> pd.DataFrame:
    """
    Converts json dataset file to a pandas dataframe of transcriptions

    :param file_path: Path to the json dataset file
    :return: Pandas dataframe of the dataset
    """
    with open(file_path, "r") as file:
        temp = pd.read_json(file, lines=True)
        temp = temp[temp["transcript"].str.len() > 0].explode("transcript")

    return pd.DataFrame(list(temp["transcript"]))


def fetch_transcript(video_id: str) -> DataFrame:
    """
    Fetches a transcript for a video id using the youtube api.
    It tries to fetch either manual or autogenerated transcripts.

    Throws exceptions specific to youtube_transcript_api if there are no transcripts or if they are disabled.
    Implement error handling when calling the function.

    :param video_id:
    :return: DataFrame of the transcript with columns "word", "start", and "ad"
    """
    reg1 = re.compile(r"\s*[(\[](\s|\w)*[])]")
    reg2 = re.compile(r"([\n\"\-,^\s._])+")

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=transcriptLang)

    processed_transcript = {"word": [], "start": [], "ad": []}

    r = requests.get(f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}")
    if r.status_code == 200:
        sponsor_segments = r.json()
        sponsor_timestamps = [
            (segment["segment"][0], segment["segment"][1])
            for segment in sponsor_segments
        ]
    else:
        sponsor_timestamps = []

    for line in transcript:
        clean_text = reg1.sub(" ", line["text"]).strip()
        clean_text = reg2.sub(" ", clean_text).strip()
        if not clean_text:
            continue

        if not sponsor_timestamps:
            is_ad = None
        else:
            is_ad = False

        line_start = line["start"]
        line_len = len(line["text"].split())

        for index, word in enumerate(line["text"].split()):
            word_start = line_start + index * line["duration"] / line_len
            word_end = word_start + line["duration"] / line_len
            for sponsor_start, sponsor_end in sponsor_timestamps:
                if (
                    sponsor_end - 1 > word_start > sponsor_start
                    and word_end > sponsor_start
                ):
                    is_ad = True
                    break

            processed_transcript["word"].append(word)
            processed_transcript["start"].append(round(word_start, 2))
            processed_transcript["ad"].append(is_ad)

    df_transcript = pd.DataFrame(processed_transcript)
    return df_transcript


if __name__ == "__main__":
    build_dataset()

