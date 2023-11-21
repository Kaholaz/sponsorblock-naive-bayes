# sponsorblock-naive-bayes
This project explores the possibility of detecting sponsor segments on streaming platforms. The primary goal for the application is speed and low resource usage, as the model is intended to be used on low-end hardware, such as personal computers and mobile devices.

The model that is explored in this project is the Naive Bayes based statistical classifier, which is repurposed and used for detecting dynamic length ads/sponsor-segments in the audio medium.

For the supervised learning of the model [SponsorBlock API](https://sponsor.ajay.app/) is used to determine ad segments. [youtube_transcript_api](https://pypi.org/project/youtube-transcript-api/) is used to fetch the transcriptions of youtube videos. In addition [WhisperX](https://github.com/m-bain/whisperX) is used to transcribe audio files locally for non-youtube videos.


## Pre-requisites

- Python 3.10+
- 16GB RAM

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Dataset and data format
The dataset must be in the format of a csv file with the following columns:

```csv
word,start,ad
```
The transcribers included in this repository output the data in this format. If you create your own transcriber, make sure to output the data in this format.

If you need to build a dataset, follow the steps in the section [Building datasets](#building-datasets).

### Transcribing
If you want to transcribe individual files, you can use the code file `transcriber.py` to get transcriptions from either YouTube videos or transcribing local audio files.

```bash
python transcriber.py [-h] (-v VIDEO | -a AUDIO_PATH) [-s SAVE_PATH]
```
Use the -h flag for detailed descriptions of the file and the different arguments.

### Training
The model can be trained with the `train.py`. The script takes two required arguments, the path to the training data and the path to the model output. To see all the arguments run the following command:

```bash
python train.py -h
```

PS: Make sure to use the same chunking/n-gram format as the model when used for prediction/evaluation. It would make little sense to use unigrams on a model that is trained on bigrams.


### Evaluating
For evaluating a model, the file `evaluate.py` can be used.

You can run the code by using the line below and filling in the necessary arguments. The arguments encapsulated by square brackets are optional, the arguments that have no brackets are required, and the arguments encapsulated by paranthesis and separated by "|" mean only one of the arguments are required.
```bash
python evaluate.py [-h] (-v VIDEO | -a AUDIO_PATH | -d DATASET_PATH) -m MODEL_PATH [-c CHUNK_WORDS] [-w WINDOW_SIZE]
                     [-ht HAM_THRESHOLD] [--stopwords STOPWORDS] [--substitution SUBSTITUTION]
```
This code can be used to evaluate the model on youtube videos, local audio files, or already existing transcription files.


### Predicting
To predict a YouTube video, run the following command:

```bash
python predict.py --video <video_url | video_id> --model <model_path>
```

The model path is the path to the model outputted by the training script.

## Building datasets
Our process of building the dataset although comparatively much faster than other alternatives, is still time consuming.
Before doing anything else, make sure to have downloaded the sponsorTimes.csv file from the SponsorBlock database, using this repository [sb-mirror](https://github.com/mchangrh/sb-mirror)
Move the sponsorTimes.csv file into this folder {PROJECT_ROOT}/transcribers/sponsor_data, if the folder doesn't exist, create it.

Once the file {PROJECT_ROOT}/transcribers/sponsor_data/sponsorTimes.csv exists, follow these steps:

### Step 1 (optional)
This step essentially filters out the worst timestamps that have very little user interaction, or are timestamped on videos with low viewcount. It also removes music videos as almost all of these don't contain any sponsor segments. It finally sorts it in descending order by viewcount, this ensures later on that one starts fetching transcriptions in the order of most views to lowest.

For only this step, you need a YouTube API key to check the video categories of the video ids in the sponsorTimes.csv file in order to filter out music videos.

Insert this API key into the filter_sort_times.py in the API_KEY variable.

Once done, run the code with:
```bash
python filter_sort_sponsor_times.py
```

### Step 2
Next step, if you've done step 1 continue as normal. Otherwise rename the file sponsorTimes.csv to processed_sponsorTimes.csv, and make sure the file {PROJECT_ROOT}/transcribers/sponsor_data/processed_sponsorTimes.csv exists. Also a word of caution, skipping step 1 will drastically lower the labelling quality of the transcriptions which will hurt the model.

Run `filter_sort_sponsor_times.py` to fetch the transcripts for the video ids that are in the processed_sponsorTimes.csv file.
This code tries fetching manual transcriptions, and if that fails, fetches autogenerated ones. These are saved into separate ndjson files for autogenerated and manual ones to preserve all transcription information. This enables us to treat each run as a continuous session. Rerunning the code simply resumes the previous session from where it left off.
```bash
python filter_sort_sponsor_times.py
```
After this step you should have two ndjson files, one for autogenerated ones and one for manual ones, in your project root folder.


### Step 3.1 (optional)
The final step consists of two stages, the first one is optional and is for fetching the timestamps using the SponsorBlock API.
The reason for this is because the timestamps in the sponsorTimes.csv file don't match the API completely.

To decide which transcriptions file it should fetch the timestamps for, change the path variable in the `transcription_dataset_json_to_csv.py` `transcription_path = transcription_dir + "youtube_manual_transcriptions.ndjson"`
If everything is in order, comment out the method call for the method get_timestamps() and run the file to fetch timestamps directly through the SponsorBlock API:
```bash
python transcription_dataset_json_to_csv.py
```
After this stage all the timestamps are saved by video id as a ndjson file in this path {PROJECT_ROOT}/transcribers/sponsor_data/sponsor_timestamps.ndjson

### Step 3.2
This final step consists of converting the ndjson file into correct csv format (word,start,ad).

If you skip Step 3.1, this stage defaults to using the timestamps we already have in processed_sponsorTimes.csv.
Comment out the method call for json_to_csv_transcripts() in the code `transcription_dataset_json_to_csv.py`, then run the file:
```bash
python transcription_dataset_json_to_csv.py
```
After this stage, you should now have a large dataset in correct csv format in the project root, which can be used to train the model.
If you haven't skipped the optional steps, the text corpus labelling quality is going to be considerably higher.

## Results

The model has shown promising results when classifying sponsor segments in YouTube videos. If this project is to be used on other content types one may or may not find success using this model, assuming one manages to accumulate a sufficiently large text corpus to train on.

We've been unable to test training the model on other content-types than YouTube videos due to time constraints, even using Speech To Text modules like WhisperX would still take 3+ minutes for a single video, making this extremely time consuming when transcribing thousands of files. This project has therefore relied mainly on YouTube's caption API for gathering transcriptions.


## Further work

- Look into wordembedding
- Create a web extension
