"""
    sponsorTimes.csv was fetched from https://sponsor.ajay.app/database ahead of time.
    The sponsorTimes.csv file contains sponsor segment timestamps for a lot of youtube videos.
    This code can be used to trim off unnecessary columns from sponsorTimes.csv,
        then sort videoID by views in descending order.
    The sponsorTimes.csv can then be used in youtube_transcription_fetcher.py to fetch the manual english
        transcript for each yt video id, and also assign each sentence in each transcript as an ad or not.
"""

import pandas as pd
chunk_size = 50000
chunks = []

directory = ".idea/"

sponsorTimes_path = directory + "sponsorTimes.csv"
processed_sponsorTimes_path = directory + "processed_sponsorTimes.csv"

for chunk in pd.read_csv(sponsorTimes_path, usecols=["videoID", "startTime", "endTime", "views"], chunksize=chunk_size):
    chunks.append(chunk)

df_combined = pd.concat(chunks)

df_sorted = df_combined.sort_values(by="views", ascending=False)

df_sorted.to_csv(processed_sponsorTimes_path, index=False)