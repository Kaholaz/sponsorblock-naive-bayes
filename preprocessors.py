import sys
import re
from typing import Optional, Callable
from typing import Optional, Callable
from pandas import DataFrame
from tqdm import tqdm
from typing import Optional, Callable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tqdm.pandas()


def substitution_preprocessor(word: str) -> str:
    """
    Preprocesses the input text by removing non-alphanumeric characters and converting it to lowercase.

    :arg text: The text to preprocess.
    :return: The preprocessed text.

    """
    clean_word = str(word)
    clean_word = re.sub(r"[^a-zA-Z\s]", "", clean_word).lower()

    return clean_word


_stopword_list = stopwords.words("english")
_word_lemmatizer = WordNetLemmatizer()


def stopword_preprocessor(word: str):
    return " ".join(
        _word_lemmatizer.lemmatize(word)
        for w in word.lower().split()
        if w not in (_stopword_list)
    )

DEFUALT_PREPROCESSORS = [substitution_preprocessor, stopword_preprocessor]
DEFAULT_CHUNK_WORDS = 1

def preprocess_words(
    text: DataFrame,  
    chunk_words: int,
    preprocessors: Optional[list[Callable[[str], str]]] = None,
) -> DataFrame:
    """
    Preprocesses by running a list of preprocessors on each word in a list of words.

    :param text: A list of unprocessesd words. Defaults to substitution_preprocessor.
    :return: A list of processed words.
    """

    print("Chunking words...")
    if chunk_words > 1:
        text["chunked_word"] = text["word"]
        text["shifted_word"] = text["word"]
        for _ in range(1, chunk_words):
            text["shifted_word"] = text["shifted_word"].shift(-1)
            text["chunked_word"] = text["chunked_word"] + " " + text["shifted_word"]

        text["word"] = text["chunked_word"]
        text.drop(columns=["chunked_word", "shifted_word"], inplace=True)
        text = text.iloc[: -chunk_words + 1]

    print("Preprocessing text...")
    clean_text = text
    for preprocessor in preprocessors:
        print(f"Running {preprocessor.__name__}...")
        clean_text["word"] = clean_text["word"].progress_apply(preprocessor)

    clean_text = clean_text.loc[clean_text["word"] != ""]
    clean_text.reset_index(drop=True, inplace=True)

    return clean_text