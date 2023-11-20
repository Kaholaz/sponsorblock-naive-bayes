# Ad-Segment-Classification-Naive-Bayes

The intent of this project was to explore the possibility of extending the option of text based ad segment detection over to (compared to deep learning models) lower-end hardware.
The model that is explored in this project is therefore a Naive Bayes based statistical classifier, which is repurposed and used for detecting dynamic length ads/sponsor-segments in the audio medium.

This model can be trained on a labeled text corpus that follow (word,start,ad) csv format.
The model has shown promising results when classifying sponsor segments in YouTube videos. If this project is to be used on other content types one may or may not find success using this model, assuming one manages to accumulate a sufficiently large text corpus to train on.
We've been unable to test training the model on other content-types than YouTube videos due to time constraints, even using Speech To Text modules like WhisperX would still take 3+ minutes for a single video, making this extremely time consuming when transcribing thousands of files. This project has therefore relied mainly on YouTube's caption API for gathering transcriptions.
The process of accumulating the training dataset consists of a combination of the [SponsorBlock API](https://sponsor.ajay.app/) and the [youtube_transcript_api](https://pypi.org/project/youtube-transcript-api/) python module to create a labelled text corpus.


## Pre-requisites

- Python 3.10+
- 16GB RAM

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Transcribing

TODO

### Training

Naive bayes is supervised learning. The format for the training data must be csv formatted as follows:

```csv
word,start,ad
```

The transcribers included in this repository output the data in this format. If you create your own transcriber, make sure to output the data in this format.

The model can be trained with the `train.py`. The script takes two required arguments, the path to the training data and the path to the model output. To see all the arguments run the following command:

```bash
python train.py -h
```

### Predicting
To predict a YouTube video, run the following command:

```bash
python predict.py --video <video_url | video_id> --model <model_path>
```

The model path is the path to the model outputted by the training script.


## Further work

- Look into wordembedding
- Web extension
