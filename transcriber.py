from pathlib import Path
import transcribers.audio_transcriber as transcriber
import argparse
from config import ROOT_DIR

SAVE_PATH = ROOT_DIR + "/transcriptions/transcription.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribes a youtube video or audio file, and saves transcription "
        "to a path."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-v",
        "--video",
        type=str,
        help="For transcribing YouTube video. Can either be a YouTube video ID or "
        "a URL.",
    )
    group.add_argument(
        "-a",
        "--audio-path",
        type=str,
        help="For transcribing a local audio file. Has to be absolute path "
        "of the audio file.",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        help="Optional full file path to save to, including filetype. If "
        "ommitted, defaults the save path to /transcriptions folder in "
        "project root.",
        required=False,
        default=SAVE_PATH,
    )

    args = parser.parse_args()

    if args.video:
        df = transcriber.get_transcription(args.video)
    elif args.audio_path:
        df = transcriber.transcribe_segment(args.audio_path, delete_file=False)
    else:
        raise Exception(
            "Missing args. Must provide something to transcribe, either --video or --audio-path"
        )

    path = Path(args.save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
