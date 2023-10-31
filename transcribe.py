import json
import yt_dlp
import whisper_timestamped as whisper
import pandas as pd
import requests
import os

def get_video_id(video_id: str) -> str:
    """
    Ensure we got a video-id.

    :param video_id: Either a video-id or an url to a YouTube video.
    :return: Returns the video-id of the YouTube video.
    """
    pattern = r'^[-_a-zA-Z0-9]{11}$'
    match = re.search(pattern, video_id)
    if match:
        return video_id

    pattern = r'[?&]v=([-_a-zA-Z0-9]{11})'
    match = re.search(pattern, video_id)
    if match:
        return match.group(1)
    else:
        raise ValueError("Neither a YouTube video-url or a video-id was supplied.")



def transcribe(video_id: str) -> pd.DataFrame:
    """
    Downloads an mp3 of a youtube video, then captions and lables it.

    :param video_id: The id of a YouTube video (the 'v' GET param), or an URL
    :return: Returns a DataFrame with the columns 'word', 'start', and 'ad'
    """
    # Ensure we've got a video-id
    video_id = get_video_id(video_id)

    try:
        yt_dlp.main(
            f"-f bestaudio -x --audio-format mp3 --audio-quality 0 -o tmp {video_id}".split(
                " "
            )
        )
    except SystemExit:
        # Ignore the sys.exit call from yt-dlp
        pass

    audio = whisper.load_audio("tmp.mp3")
    os.remove("tmp.mp3")
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")

    words = []
    starts = []
    ads = []
    for o in result["segments"]:
        for word in o["words"]:
            words.append(word["text"])
            starts.append(word["start"])
            ads.append(False)

    df = pd.DataFrame({"word": words, "start": starts, "ad": ads})
    r = requests.get(
        f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}")
    if r.status_code == 404:
        # Not in the database -> no ads.
        return df

    segments = [s["segment"]
                for s in r.json()]
    if len(segments) == 0:
        return df

    segment_i = 0
    segment = segments[segment_i]
    for df_i, row in df.iterrows():
        # Before an ad
        if row["start"] < segment[0]:
            continue

        # After an ad
        if row["start"] > segment[1]:
            segment_i += 1
            if segment_i >= len(segments):
                break  # No more ads
            segment = segments[segment_i]
            continue

        df.at[df_i, "ad"] = True

    return df
