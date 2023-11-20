# SponsorBlock-naive-bayes

Automatically detect sponsor segments in YouTube videos using a Naive Bayes classifier. The training process utilizes the [SponsorBlock API](https://sponsor.ajay.app/). Naive bayes is a simple classifier able to run on low-end hardware. The model is supposed to run on your local machine, with no cost.

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
