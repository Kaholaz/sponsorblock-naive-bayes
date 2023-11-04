"""
    sponsorTimes.csv was fetched from https://sponsor.ajay.app/database ahead of time.
    The sponsorTimes.csv file contains sponsor segment timestamps for a lot of youtube videos.
    This code can be used to trim off unnecessary columns from sponsorTimes.csv,
        then sort videoID by views in descending order.
    It also filters out certain rows depending on category, and number of votes.
    This ensures that only sponsor categories are included, and that segments are somewhat verified by users.
    It also checks video category using the youtube API to filter out music videos as well.
    The processed sponsorTimes.csv can then be used in youtube_transcription_fetcher.py to fetch the english
        transcripts for each yt video id, and also assign each sentence in each transcript as an ad or not.
"""

import pandas as pd
import requests

chunk_size = 50000
chunks = []

directory = ".idea/"

sponsorTimes_path = directory + "sponsorTimes.csv"
processed_sponsorTimes_path = directory + "processed_sponsorTimes.csv"

API_KEY = "{Insert_Youtube_API_Key}"
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"


def get_categories(video_ids):
    max_ids_per_request = 50
    video_categories = {}

    for i in range(0, len(video_ids), max_ids_per_request):
        batch_ids = video_ids[i:i+max_ids_per_request]
        params = {
            "part": "snippet",
            "id": ",".join(batch_ids),
            "key": API_KEY,
            "fields": "items(id,snippet(categoryId))"
        }
        response = requests.get(YOUTUBE_API_URL, params=params)
        response.raise_for_status()
        if i%1000 == 0:
            print(f"Processed {i} videoIDs")

        for item in response.json().get("items", []):
            video_categories[item["id"]] = item["snippet"]["categoryId"]

    return video_categories


for chunk in pd.read_csv(sponsorTimes_path, usecols=["videoID", "startTime", "endTime", "views", "votes", "category"], chunksize=chunk_size):
    filtered_chunk = chunk[(chunk["votes"] >= 3) & (chunk["category"] == "sponsor")]
    chunks.append(filtered_chunk)

df_combined = pd.concat(chunks)

df_sorted = df_combined.sort_values(by="views", ascending=False)

unique_video_ids = df_sorted["videoID"].unique().tolist()

video_categories = get_categories(unique_video_ids)

music_videos = {video_id for video_id, category_id in video_categories.items() if category_id == "10"}

df_sorted = df_sorted[~df_sorted["videoID"].isin(music_videos)]

df_sorted.to_csv(processed_sponsorTimes_path, index=False)

print(f"Filtered out video ids classified as music, total {len(music_videos)} videos:\n{music_videos}")