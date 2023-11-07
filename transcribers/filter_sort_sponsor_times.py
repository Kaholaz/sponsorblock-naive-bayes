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
from config import ROOT_DIR
import pandas as pd
import requests
import os

chunk_size = 50000
chunks = []

directory = ROOT_DIR + "/dataset_builder/sponsor_data/"
if not os.path.exists(directory):
    os.makedirs(directory)

sponsorTimes_path = directory + "sponsorTimes.csv"
processed_sponsorTimes_path = directory + "processed_sponsorTimes.csv"

API_KEY = "{Insert_Youtube_API_Key}"
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"


def get_categories(video_ids: list) -> dict[str, str]:
    max_ids_per_request = 50
    video_categories = {}

    for i in range(0, len(video_ids), max_ids_per_request):
        batch_ids = video_ids[i:i + max_ids_per_request]
        params = {
            "part": "snippet",
            "id": ",".join(batch_ids),
            "key": API_KEY,
            "fields": "items(id,snippet(categoryId))"
        }
        response = requests.get(YOUTUBE_API_URL, params=params)
        response.raise_for_status()
        if i % 1000 == 0:
            print(f"Processed {i} videoIDs")

        for item in response.json().get("items", []):
            video_categories[item["id"]] = item["snippet"]["categoryId"]

    return video_categories


if __name__ == "__main__":
    try:
        for chunk in pd.read_csv(sponsorTimes_path, usecols=["videoID", "startTime", "endTime", "views", "votes", "category"],
                                 chunksize=chunk_size):
            filteredChunk = chunk[(chunk["votes"] >= 3) & (chunk["category"] == "sponsor")]
            chunks.append(filteredChunk)
    except FileNotFoundError as e:
        print(f"No sponsorTime.csv data file was found: {e}")
        exit(1)

    dfCombined = pd.concat(chunks)

    dfSorted = dfCombined.sort_values(by="views", ascending=False)

    uniqueVideoIds = dfSorted["videoID"].unique().tolist()

    videoCategories = get_categories(uniqueVideoIds)

    musicVideos = {video_id for video_id, category_id in videoCategories.items() if category_id == "10"}

    dfSorted = dfSorted[~dfSorted["videoID"].isin(musicVideos)]

    dfSorted.to_csv(processed_sponsorTimes_path, index=False)

    print(f"Filtered out video ids classified as music, total {len(musicVideos)} videos:\n{musicVideos}")
