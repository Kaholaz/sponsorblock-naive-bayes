import pandas as pd
chunk_size = 50000
chunks = []

directory = ".idea"

sponsorTimes_path = directory + "sponsorTimes.csv"
processed_sponsorTimes_path = directory + "processed_sponsorTimes.csv"

for chunk in pd.read_csv(sponsorTimes_path, usecols=["videoID", "startTime", "endTime", "views"], chunksize=chunk_size):
    chunks.append(chunk)

df_combined = pd.concat(chunks)

df_sorted = df_combined.sort_values(by="views", ascending=False)

df_sorted.to_csv(processed_sponsorTimes_path, index=False)