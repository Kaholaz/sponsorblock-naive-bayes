import transcribers.audio_transcriber as transcriber
import argparse
from config import ROOT_DIR

SAVE_PATH = ROOT_DIR+"/transcriptions/transcription.csv"

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Transcribes a youtube video or audio file, and saves transcription to a path.")

    argparse.add_argument("--video", type=str, help="For transcribing YouTube video. Can either be a YouTube video ID or a URL.", required=False)
    argparse.add_argument("--audio_path", type=str, help="For transcribing a local audio file. Has to be absolute path of the audio file.", required=False)
    argparse.add_argument("--save_path", type=str, help="Optional full file path to save to, including filetype. If ommitted, defaults the save path to /transcriptions folder in project root.", required=False, default=SAVE_PATH)

    args = argparse.parse_args()

    if args.video:
        df = transcriber.get_transcription(args.video)
    elif args.audio_path:
        df = transcriber.transcribe_segment(args.audio_path, delete_file=False)
    else:
        raise Exception("Missing args. Must provide something to transcribe, either --video or --audio_path")

    df.to_csv(args.save_path, index=False)
