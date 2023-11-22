"""
    After processing sponsorTimes.csv and fetching the transcripts for a specified number of videoIDs.
        in processed_sponsorTimes.csv file with the youtube_transcription_fetcher.py script and building
        the json dataset, this script is used to convert the format to (word, start, ad).
    It also verifies the ad labels and readjusts the ad labels from sentence level to word level.
"""
from pandas import DataFrame
from config import ROOT_DIR
import pandas as pd
import requests
import json
import re

transcription_dir = ROOT_DIR
transcription_path = transcription_dir + "youtube_manual_transcriptions.ndjson"
processed_sponsorTimes_path = "sponsor_data/processed_sponsorTimes.csv"

transcripts = pd.read_json(transcription_path, lines=True)


def get_timestamps() -> None:
    """
    Gets sponsorblock timestamps for each video ID using the sponsor block api and saves them to a file.
    This is done because dataset timestamps gotten from the sponsorblock database are not always accurate
        or identical to its api for some reason.
    These fetched timestamps can be used to re-verify and fix the ad labels in the dataset while converting
        to csv with format (word, start, ad).

    :return:
    """
    with open("sponsor_data/sponsor_timestamps.ndjson", "w") as file:
        for _, row in transcripts.iterrows():
            video_id = row["video_id"]
            r = requests.get(
                f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}&category=sponsor"
            )
            if r.status_code != 404:
                sponsor_segments = r.json()
                sponsor_timestamps = [
                    (segment["segment"][0], segment["segment"][1])
                    for segment in sponsor_segments
                ]
                print(sponsor_timestamps)
            else:
                print(
                    f"{video_id}--------------------{r.status_code}-------------------",
                    r.text,
                )
                continue

            sponsor_segment_timestamps = {
                "video_id": video_id,
                "sponsor_times": sponsor_timestamps,
            }
            file.write(json.dumps(sponsor_segment_timestamps) + "\n")


def json_to_csv_transcripts(_transcripts: DataFrame, _sponsor_times: DataFrame) -> None:
    """
    Converts the json dataset to csv format (word, start, ad) and adjusts the ad labels.
    If you've fetched the sponsor timestamps from the api, those will be used whenever possible.
    Otherwise, it defaults to the sponsor timestamps from the _sponsor_times dataframe,
        which is read from the processed_sponsorTimes.csv file.

    :param _transcripts: The youtube transcription json dataset read into a DataFrame
    :param _sponsor_times: The processed_sponsorTimes.csv read into a DataFrame
    :return:
    """
    print("Reformatting to csv and relabeling transcripts")
    sponsor_segments_from_api = None
    try:
        sponsor_segments_from_api = pd.read_json(
            "sponsor_data/sponsor_timestamps.ndjson", lines=True
        )
        sponsor_segments_from_api.set_index("video_id", inplace=True)
    except FileNotFoundError:
        print(
            "sponsor_timestamps.ndjson not found, using only timestamps from _sponsor_times"
        )

    sponsor_segments_database = _sponsor_times

    csv_transcript = {"word": [], "start": [], "ad": []}

    for _, row in _transcripts.iterrows():
        sponsor_segments = []
        if sponsor_segments_from_api is not None:
            try:
                sponsor_segments = sponsor_segments_from_api.at[
                    row["video_id"], "sponsor_times"
                ]
                # print(sponsor_segments)
            except KeyError:
                print(
                    f"sponsor_segments_from_api does not have timestamps for {row['video_id']} using processed_sponsorTimes.csv timestamps"
                )

        if len(sponsor_segments) == 0:
            sponsor_segments = sponsor_segments_database[
                sponsor_segments_database["videoID"] == row["video_id"]
            ][["startTime", "endTime"]].to_numpy()

        # print(sponsor_segments, "  ", row["video_id"])

        for transcript in row["transcript"]:
            words = [word for word in re.split(r"\s+", transcript["text"]) if word]
            sentence_start = transcript["start"]
            average_duration = transcript["duration"] / len(words)

            for index, word in enumerate(words):
                word_start = sentence_start + index * average_duration
                word_end = word_start + average_duration

                is_ad = False
                for sponsor in sponsor_segments:
                    sponsor_start, sponsor_end = sponsor
                    if (
                        sponsor_end > word_start > sponsor_start
                        and word_end > sponsor_start
                    ):
                        is_ad = True
                        break

                csv_transcript["word"].append(word)
                csv_transcript["start"].append(round(word_start, 2))
                csv_transcript["ad"].append(is_ad)

    print("Finished reformatting and readjusting labels, saving to csv")
    pd.DataFrame(csv_transcript).to_csv(
        transcription_dir + "youtube_manual_transcriptions.csv", index=False
    )


# get_timestamps()

# json_to_csv_transcripts(transcripts, pd.read_csv(processed_sponsorTimes_path))
