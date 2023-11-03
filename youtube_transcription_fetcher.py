import json
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import pandas as pd
import os
import re

processed_sponsorTimes_path = ".idea/processed_sponsorTimes.csv"

transcription_dir = "transcriptions/"
os.makedirs(transcription_dir, exist_ok=True)

transcription_path = transcription_dir + "youtube_transcriptions.json"
no_transcript_path = transcription_dir + "yt_vids_without_manual_english_transcript.csv"

transcript_language = ["en-GB", "en-US", "en-CA", "en-AU", "en-NZ", "en"]

try:
    df_no_transcript = pd.read_csv(no_transcript_path)
    no_transcript_set = set(df_no_transcript["videoID"])
except FileNotFoundError:
    no_transcript_set = set()
    pd.DataFrame(columns=["videoID"]).to_csv(no_transcript_path, index=False)

df_video_sponsor_data = pd.read_csv(processed_sponsorTimes_path, nrows=50000)

for _, row in df_video_sponsor_data.iterrows():
    video_id = row["videoID"]
    start_time = row["startTime"]
    end_time = row["endTime"]

    with open(transcription_path, "r") as file:
        if video_id in file.read():
            print(f"Transcript for videoID {video_id} already exists")
            continue

    with open(no_transcript_path, "r") as file:
        if video_id in file.read() or video_id in no_transcript_set:
            print(f"Skipping videoID {video_id}, as no transcript existed when checked during a past run")
            continue

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_manually_created_transcript(transcript_language).fetch()

        processed_transcript = {
            "video_id": video_id,
            "transcript": []
        }
        for line in transcript:
            clean_text = re.sub(r"([\n\"\-,^\s])+", " ", line["text"]).strip()
            is_ad = start_time <= line["start"] <= end_time
            processed_transcript["transcript"].append({
                "text": clean_text,
                "start": line["start"],
                "duration": line["duration"],
                "ad": is_ad
            })

        with open(transcription_path, "a") as json_file:
            json_file.write(json.dumps(processed_transcript) + "\n")

    except NoTranscriptFound:
        print(f"Manual transcript not found for video ID {video_id}")
        no_transcript_set.add(video_id)
        with open(no_transcript_path, "a") as csv_file:
            csv_file.write(video_id + "\n")

    except Exception as e:
        print(f"An error occurred for video ID {video_id}: {e}")


transcript_data = []

with open(transcription_path, "r") as file:
    for line in file:
        record = json.loads(line)
        for transcript in record["transcript"]:
            transcript_data.append({
                "text": transcript["text"],
                "start": transcript["start"],
                "duration": transcript["duration"],
                "ad": transcript["ad"]
            })

df_transcripts = pd.DataFrame(transcript_data)

print(df_transcripts)