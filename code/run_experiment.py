#!/usr/bin/env python3
import argparse
import json
import os
import random
import string

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MAX_LOG_LEVEL"] = "3"

# import emoji
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from ConfigSpace import ConfigurationSpace, hyperparameters as CH
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import MaxAbsScaler
from smac.facade.smac_hpo_facade import SMAC4HPO as FacadeOptimizer
from smac.scenario.scenario import Scenario
from tensorflow.keras import callbacks, layers as L, Model, optimizers
from transformers import RobertaTokenizer, TFRobertaModel


# {{{ defaults
DEFAULT_OUTPUT_DIR = "/outputs/chatgpt_evaluations/iteration_3/"  # "outputs"
reference = {
    "sarcasm": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.8,
        "batch_size": 256,
    },
    "subjectivity": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.8,
        "batch_size": 128,
    },
    "suicide": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.85,
        "batch_size": 256,
    },
    "sentiment": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.75,
        "batch_size": 512,
    },
    "personality": {
        "loss_fn": "mean_absolute_error",
        "final_activation": "sigmoid",
        "monitor_metric": "mean_absolute_error",
        "baseline": 0.12,
        "batch_size": 64,
        "num_features": 5,
    },
    "toxicity": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.78,
        "num_features": 6,
        "batch_size": 512,
        "use_weight": True,
    },
    "sentiment_ranking_microblogs": {
        "loss_fn": "mean_absolute_error",
        "final_activation": "tanh",
        "monitor_metric": "mean_absolute_error",
        "baseline": 0.30,
        "batch_size": 8,
    },
    "sadness_emotion_intensity": {
        "loss_fn": "mean_absolute_error",
        "final_activation": "sigmoid",
        "monitor_metric": "mean_absolute_error",
        "baseline": 0.14,
        "batch_size": 8,
    },
    "joy_emotion_intensity": {
        "loss_fn": "mean_absolute_error",
        "final_activation": "sigmoid",
        "monitor_metric": "mean_absolute_error",
        "baseline": 0.14,
        "batch_size": 8,
    },
    "fear_emotion_intensity": {
        "loss_fn": "mean_absolute_error",
        "final_activation": "sigmoid",
        "monitor_metric": "mean_absolute_error",
        "baseline": 0.14,
        "batch_size": 8,
    },
    "anger_emotion_intensity": {
        "loss_fn": "mean_absolute_error",
        "final_activation": "sigmoid",
        "monitor_metric": "mean_absolute_error",
        "baseline": 0.14,
        "batch_size": 8,
    },
    "well_being_reddit_body": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.75,
        "batch_size": 32,
    },
    "well_being_reddit": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.8,
        "batch_size": 32,
    },
    "well_being_reddit_titles": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.8,
        "batch_size": 16,
    },
    "well_being_twitter": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.8,
        "batch_size": 16,
    },
    "well_being_twitter_full": {
        "loss_fn": "binary_crossentropy",
        "final_activation": "sigmoid",
        "monitor_metric": "uar",
        "baseline": 0.8,
        "batch_size": 16,
    },
    "engagement": {
        "loss_fn": "mean_absolute_error",
        "final_activation": "relu",
        "monitor_metric": "mean_absolute_error",
        "baseline": 0.4,
        "batch_size": 256,
    },
    "aspect_res14_target": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "target",
        "baseline": 0.5,
        "num_features": 3,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_lap14_target": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "target",
        "baseline": 0.5,
        "num_features": 3,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_res15_target": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "target",
        "baseline": 0.5,
        "num_features": 3,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_res14_polarity": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "target_polarity",
        "baseline": 0.45,
        "num_features": 5,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_lap14_polarity": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "target_polarity",
        "baseline": 0.45,
        "num_features": 5,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_res15_polarity": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "target_polarity",
        "baseline": 0.45,
        "num_features": 5,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_res14_opinion": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "opinion",
        "baseline": 0.5,
        "num_features": 5,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_lap14_opinion": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "opinion",
        "baseline": 0.5,
        "num_features": 3,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
    "aspect_res15_opinion": {
        "loss_fn": "sparse_categorical_crossentropy",
        "final_activation": "softmax",
        "monitor_metric": "uar",
        "labels": "opinion",
        "baseline": 0.5,
        "num_features": 3,
        "batch_size": 64,
        "time_distributed": True,
        "use_weight": True,
    },
}
# }}}


# {{{ Validation functions

def validate_types(lst, *types):
    return any(all(isinstance(x, typ) for x in lst) for typ in types)


def validate_types_2d(lst, *types):
    return any(all(validate_types(x, typ) for x in lst) for typ in types)


def validate(fn):

    def func(part):
        assert part in ["train", "valid", "test"], "Invalid part"
        results = fn(part)
        unique = set([len(res) for res in results]) 
        assert len(unique) == 1, "All results should have the same length, %s" % unique
        if part == "test":
            assert len(results) >= 2, "Expected at least columns: texts, labels, ChatGPT"
        else:
            assert len(results) == 2, "Expected 2 columns: texts, labels"
        assert validate_types(results[0], str), "First column should be texts"
        assert validate_types(results[1], int, float) or validate_types_2d(results[1], int, float), "Second column should be int/float"

        return results

    func.__name__ = fn.__name__
    return func

# }}}


# {{{ Util functions

def filter_long_texts(df, column='text'):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    mask = df[column].map(lambda x: isinstance(x, str) and 2 <= len(tokenizer.tokenize(x)) < 510)
    return df[mask]


def cols_to_lists(df, *cols):
    return tuple([df[col].values.tolist() for col in cols])


def chunk_items(items, size=32):
    for start in range(0, len(items), size):
        sub = items[start: start + size]
        yield tuple(zip(*sub))
        # X, Y = zip(*sub)
        # yield X, Y
    return

# }}}


# {{{ Reading functions for each problem

@validate
def read_sarcasm(part):
    test_df = pd.read_csv("chatgpt-responses/sarcasm.csv", index_col=0)
    if part == "test":
        test_df = filter_long_texts(test_df, "headline")
        return cols_to_lists(test_df, "headline", "is_sarcastic")  # , "ChatGPT")
        # return cols_to_lists(test_df, "headline", "is_sarcastic", "ChatGPT")

    lns = [json.loads(x) for x in open("datasets/sarcasm identification/data/Sarcasm_Headlines_Dataset.json").read().split('\n')[:-1]]
    sarcasm_df = pd.DataFrame.from_records(lns)
    df = sarcasm_df.sample(frac=1., random_state=41).reset_index()
    N = test_df.shape[0]
    assert np.all(df["index"].values[:N] == test_df["index"].values)
    df = df[N:]
    V = 4000
    df = filter_long_texts(df, "headline")
    train_df, valid_df = df[V:], df[:V]
    if part == "train":
        return cols_to_lists(train_df, "headline", "is_sarcastic")
    elif part == "valid":
        return cols_to_lists(valid_df, "headline", "is_sarcastic")


def read_emotion_intensity(emotion, part):
    df = pd.read_csv(f"datasets/intensity ranking/emotion intensity/{part}/{emotion}.txt", delimiter="\t", names=["id", "text", "emotion", "score"])
    df = filter_long_texts(df, "text")
    return cols_to_lists(df, "text", "score")


@validate
def read_anger_emotion_intensity(part):
    return read_emotion_intensity("anger", part)


@validate
def read_joy_emotion_intensity(part):
    return read_emotion_intensity("joy", part)


@validate
def read_sadness_emotion_intensity(part):
    return read_emotion_intensity("sadness", part)


@validate
def read_fear_emotion_intensity(part):
    return read_emotion_intensity("fear", part)


@validate
def read_subjectivity(part):
    test_df = pd.read_csv("chatgpt-responses/subjectivity.csv")
    subjective = open("datasets/subjectivity detection/rotten_imdb/quote.tok.gt9.5000", encoding="ISO-8859-1").read().split('\n')
    objective = open("datasets/subjectivity detection/rotten_imdb/plot.tok.gt9.5000", encoding="ISO-8859-1").read().split('\n')

    df = pd.concat([
        pd.DataFrame({"text": subjective, "label": 1}),
        pd.DataFrame({"text": objective, "label": 0}),
    ]).sample(frac=1, random_state=41).reset_index()
    assert df[:2000].text.tolist() == test_df.text.tolist()
    test_df = filter_long_texts(test_df, "text")
    df = filter_long_texts(df, "text")
    valid_df = df[2000:4000]
    train_df = df[4000:]
    if part == "train":
        return cols_to_lists(train_df, "text", "label")
    elif part == "valid":
        return cols_to_lists(valid_df, "text", "label")
    elif part == "test":
        return cols_to_lists(test_df, "text", "label")  # , "ChatGPT")
        # return cols_to_lists(test_df, "text", "label", "ChatGPT")


@validate
def read_toxicity(part):
    if part == "test":
        toxicity_df = pd.read_csv("datasets/toxicity detection/data/test.csv").set_index("id")
        toxicity_df = toxicity_df.join(pd.read_csv("datasets/toxicity detection/data/test_labels.csv").set_index("id"))
        toxicity_df = toxicity_df.reset_index()
        toxicity_df = toxicity_df[~np.any(toxicity_df[toxicity_df.columns[2:]] == -1, axis=1)]
        # msk = np.any(toxicity_df[toxicity_df.columns[1:]] == 1, axis=1)
        msk = toxicity_df[toxicity_df.columns[2:]].astype(np.float32)
        weights = msk * np.power((1 - msk).sum(axis=0) / msk.sum(axis=0) + 1, 1.1)
        weights = (weights * 0.5).mean(axis=1) + 1
        # ##weights = (weights * 2).sum(axis=1) + 1
        # ##weights = msk.astype(np.float32) * 4 + 1
        dq = pd.concat([toxicity_df[c].value_counts() for c in toxicity_df.columns[2:]], axis=1)
        print(dq.sort_index())
        df = toxicity_df.sample(frac=1, random_state=41, weights=weights).reset_index()[:1000]
        dq = pd.concat([df[c].value_counts() for c in df.columns[3:]], axis=1)
        print(dq.sort_index())
        print(df.shape, df.columns[3:])
        df = filter_long_texts(df, "comment_text")
        dq = pd.concat([df[c].value_counts() for c in df.columns[3:]], axis=1)
        print(dq.sort_index())
        print(df.shape, df.columns[3:])
        return cols_to_lists(df, "comment_text", df.columns[3:])
    else:
        toxicity_df = pd.read_csv("datasets/toxicity detection/data/train.csv")
        toxicity_df = filter_long_texts(toxicity_df, "comment_text")
        msk = None
        for c in toxicity_df.columns[2:]:
            if msk is None:
                msk = (toxicity_df[c] == 1)
            else:
                msk = msk | (toxicity_df[c] == 1)
        toxicity_df = pd.concat([
            toxicity_df[msk],
            toxicity_df[~msk].sample(frac=0.15, random_state=41),  # downsample the negative class
        ], axis=0)

        dq = pd.concat([toxicity_df[c].value_counts() for c in toxicity_df.columns[2:]], axis=1)
        print(dq.sort_index())
        df = toxicity_df.sample(frac=1, random_state=41).reset_index()
        train_df, valid_df = df[:30_000], df[30_000:]
        dq = pd.concat([train_df[c].value_counts() for c in train_df.columns[3:]], axis=1)
        print(dq.sort_index())
        dq = pd.concat([valid_df[c].value_counts() for c in valid_df.columns[3:]], axis=1)
        print(dq.sort_index())

        print(df.shape, df.columns[3:])
        # return df
        if part == "train":
            return cols_to_lists(train_df, "comment_text", df.columns[3:])
        elif part == "valid":
            return cols_to_lists(valid_df, "comment_text", df.columns[3:])


def read_well_being(part, path, test_path, test_size, valid_size, text_col, labels_col):
    df = pd.read_excel(f"datasets/well-being assessment/data/{path}")
    df = df.sample(frac=1, random_state=41).reset_index()
    print(df.shape, df.columns)
    test_df = pd.read_csv(f"chatgpt-responses/{test_path}")[:test_size]
    assert df[:test_size][text_col].tolist() == test_df[text_col].tolist()

    if part == "test":
        df = test_df[:test_size]
    elif part == "valid":
        df = df[test_size: test_size + valid_size]
    elif part == "train":
        df = df[test_size + valid_size:]
    df = filter_long_texts(df, text_col)
    if part == "test":
        return cols_to_lists(df, text_col, labels_col)  # , "ChatGPT")
        # return cols_to_lists(df, text_col, labels_col, "ChatGPT")
    else:
        return cols_to_lists(df, text_col, labels_col)


@validate
def read_well_being_twitter_full(part):
    return read_well_being(part, "Twitter_Full.xlsx", "well_being_twitter_full.csv", 1500, 1500, "text", "labels")


@validate
def read_well_being_twitter(part):
    return read_well_being(part, "Twitter_Non-Advert.xlsx", "well_being_twitter.csv", 800, 400, "text", "label")


@validate
def read_well_being_reddit(part):
    return read_well_being(part, "Reddit_Combi.xlsx", "well_being_reddit.csv", 1000, 500, "title", "label")


@validate
def read_well_being_reddit_body(part):
    return read_well_being(part, "Reddit_Combi.xlsx", "well_being_reddit_body.csv", 1000, 500, "body", "label")


@validate
def read_well_being_reddit_titles(part):
    return read_well_being(part, "Reddit_Title.xlsx", "well_being_reddit_titles.csv", 1000, 1000, "title", "label")


@validate
def read_engagement(part):
    engagement_df = pd.read_csv("datasets/engagement measurement/TEDTalks.csv")
    engagement_df = engagement_df.sample(frac=1, random_state=41).reset_index()
    engagement_df['logRetweets'] = np.log10(engagement_df['retweetCount'] + 1)
    test_df = engagement_df[:4000]
    valid_df = engagement_df[4000: 9000]
    train_df = engagement_df[9000:]
    if part == "test":
        test_df = filter_long_texts(test_df, 'content')
        return cols_to_lists(test_df, "content", "logRetweets")
    elif part == "valid":
        valid_df = filter_long_texts(valid_df, 'content')
        return cols_to_lists(valid_df, "content", "logRetweets")
    elif part == "train":
        train_df = filter_long_texts(train_df, 'content')
        return cols_to_lists(train_df, "content", "logRetweets")


def read_aspect(lap, labels, part):
    part = "dev" if part == "valid" else part
    sentences = open(f"datasets/aspect extraction/data/{lap}/{part}/sentence.txt").read().split("\n")
    targets = open(f"datasets/aspect extraction/data/{lap}/{part}/{labels}.txt").read().split("\n")
    if not sentences[-1] or not targets[-1]:
        sentences = sentences[:-1]
        targets = targets[:-1]
    # targets = open(f"datasets/aspect extraction/data/{lap}/{part}/target.txt").read().split("\n")
    # opinions = open(f"datasets/aspect extraction/data/{lap}/{part}/opinion.txt").read().split("\n")
    targets = [list(map(int, t.split())) for t in targets if t]

    return sentences, targets


@validate
def read_aspect_res14_polarity(part):
    return read_aspect("res14", "target_polarity", part)


@validate
def read_aspect_lap14_polarity(part):
    return read_aspect("lap14", "target_polarity", part)


@validate
def read_aspect_res15_polarity(part):
    return read_aspect("res15", "target_polarity", part)


@validate
def read_aspect_res14_target(part):
    return read_aspect("res14", "target", part)


@validate
def read_aspect_lap14_target(part):
    return read_aspect("lap14", "target", part)


@validate
def read_aspect_res15_target(part):
    return read_aspect("res15", "target", part)


@validate
def read_aspect_res14_opinion(part):
    return read_aspect("res14", "opinion", part)


@validate
def read_aspect_lap14_opinion(part):
    return read_aspect("lap14", "opinion", part)


@validate
def read_aspect_res15_opinion(part):
    return read_aspect("res15", "opinion", part)


@validate
def read_personality(part):
    cols = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    df = pd.read_json(f"datasets/personality assessment/FI-{part}.json", orient="index")
    df = filter_long_texts(df, 'text')
    if part == "test":
        df.to_csv("personality.csv")
    return cols_to_lists(df, "text", cols)


@validate
def read_suicide(part):
    df = pd.read_json(f"datasets/suicide detection/suicide_data_{part}.json", orient="index")
    siz = {"train": 25000, "valid": 6000, "test": 2500}[part]
    df = df.reset_index().sample(frac=1., random_state=41)[:siz]
    df = filter_long_texts(df, 'text')
    df['class'] = (df['class'] == 'suicide').astype(np.int32)
    if part == "test":
        df.to_csv("suicide.csv")
    return cols_to_lists(df, "text", 'class')


@validate
def read_sentiment(part):
    df = pd.read_json(f"datasets/sentiment analysis/sentiment-{part}.json", orient="index")
    siz = {"train": 100000, "valid": 10000, "test": 2500}[part]
    df = df.reset_index().sample(frac=1., random_state=41)[:siz]
    df = filter_long_texts(df, 'text')
    df['sentiment'] = df['sentiment'] / 2
    if part == "test":
        df.to_csv("sentiment.csv")
    return cols_to_lists(df, "text", 'sentiment')


@validate
def read_sentiment_ranking_microblogs(part):
    df = pd.read_json("datasets/intensity ranking/sentiment intensity/Microblog_Trainingdata.json")
    df['spans'] = df['spans'].apply(lambda x: ' '.join(x))
    df = df.rename(columns={"sentiment score": "sentiment"})
    df = df.sample(frac=1., random_state=41)
    df = filter_long_texts(df, 'spans')
    df[1300:].to_csv("sentiment_intensity.csv")
    if part == "train":
        return cols_to_lists(df[:1000], "spans", "sentiment")
    elif part == "valid":
        return cols_to_lists(df[1000:1300], "spans", "sentiment")
    elif part == "test":
        return cols_to_lists(df[1300:], "spans", "sentiment")
# }}}


class CountVectorizer:
    def __init__(self, max_features=2000, simple_tokens=False):
        self.freq = {}
        self.index = []
        self.vocab = []
        self.simple_tokens = simple_tokens
        self.padding_token_id = 0
        self.unk_token_id = 1
        if not simple_tokens:
            self.start_seq_token_id = 2
            self.end_seq_token_id = 3
            self.max_words = max_features - 4
        else:
            self.max_words = max_features - 2
        self.stop_words = set(stopwords.words('english') + list(string.punctuation))

    def serialize(self):
        return {k: getattr(self, k) for k in ["index", "vocab", "freq", "max_words", "simple_tokens"]}

    def save(self, path):
        with open(path, "w") as fl:
            json.dump(self.serialize(), fl)

    @classmethod
    def load(cls, path):
        with open(path, "r") as fl:
            kwargs = json.load(fl)
        obj = cls()
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def _tokenize(self, sent):
        if self.simple_tokens:
            return sent.lower().split()
        else:
            return [w for w in word_tokenize(sent.lower()) if w not in self.stop_words]

    def fit(self, texts):
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                f = self.freq.get(word, 0)
                self.freq[word] = 1 + f
        pairs = sorted([(freq, word) for word, freq in self.freq.items()], reverse=True)
        pairs = pairs[:self.max_words]
        self.vocab = [word for _, word in pairs]
        self.index = {word: idx for idx, word in enumerate(self.vocab)}
        return self

    def transform(self, texts, padding=False):
        results = []
        for text in texts:
            enc = [self.index.get(word, self.unk_token_id) for word in self._tokenize(text)]
            if self.simple_tokens:
                results.append(enc)
            else:
                results.append([self.start_seq_token_id] + enc + [self.end_seq_token_id])
        if padding:
            max_length = max(len(res) for res in results)
            return [res + [self.padding_token_id] * (max_length - len(res)) for res in results]
        else:
            return results

    def fit_transform(self, texts, padding=False):
        return self.fit(texts).transform(texts, padding=padding)


def align_labels(tokenizer, sentence, word_labels, background_label=0):
    subwords = tokenizer.tokenize(sentence)
    #words = sentence.split()
    #_ids = tokenizer(sentence)["input_ids"]

    #assert len(_ids) == len(subwords) + 2 and _ids[0] == 0 and _ids[-1] == 2, (subwords, _ids)
    #assert len(words) == len(word_labels), (sentence, sentence.split(), word_labels)

    labels = [background_label]
    word_index = 0
    inv = []

    for j, subword in enumerate(subwords):
        # if the subword is actually a new word
        if j == 0 or subword.startswith("Ġ"):
            #if subword.startswith("Ġ"):
            #    subword = subword[1:]  # remove the Ġ character
            #assert words[word_index].startswith(subword), (subword, words[word_index])
            word_index += 1
            inv.append(j + 1)
        labels.append(word_labels[word_index - 1])
    labels.append(background_label)
    return labels, inv


def read_data(problem, part):
    data = globals()[f"read_{problem}"](part)
    data = list(zip(*data))  # Zip the examples together for shuffling
    if part != "test":
        random.seed(41)
        random.shuffle(data)
    print(len(data), "filtered examples...")
    return data


def generate_examples(problem, part, features):
    data = read_data(problem, part)
    time_distributed = reference[problem].get("time_distributed", False)

    if features == "RoBERTa":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        assert tokenizer.pad_token_id == 1, "Assumed like this in loading tfrecords"
        model = TFRobertaModel.from_pretrained("roberta-base")

        chunk_size = 64
        batches = chunk_items(data, chunk_size)
        total = (len(data) + chunk_size - 1) // chunk_size

        for X, outputs in tqdm(batches, total=total, ascii="░▒▓▅", ncols=80):
            X_tokens = tokenizer(X, return_tensors="tf", padding=True)
            texts = model(X_tokens).last_hidden_state.numpy()
            # if time_distributed:
            #     # Time series predictions, no pooling
            #     texts = model(X_tokens).last_hidden_state.numpy()
            # else:
            #     # Single label prediction, reduce the time dimension with pooling
            #     texts = model(X_tokens).pooler_output.numpy()
            for text, output, tokens, raw_text in zip(texts, outputs, X_tokens["input_ids"], X):
                # sequential label
                pad_mask = tokens.numpy() != tokenizer.pad_token_id
                # tokens = tokens[pad_mask]
                text = text[pad_mask]
                if time_distributed:
                    routput, inverse = align_labels(tokenizer, raw_text, output)
                    if part == "test":
                        yield text, routput, inverse, output
                    else:
                        yield text, routput
                else:
                    if np.ndim(output) < 1:
                        output = np.expand_dims(output, axis=-1)
                    yield text, output
    elif features == "BoW":
        path = os.path.join(
            DEFAULT_OUTPUT_DIR, "encoders", f"{problem}_{features}.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            print(f"{path} already exists...")
            vectorizer = CountVectorizer.load(path)
        else:
            assert part == "train"
            vectorizer = CountVectorizer(max_features=2000, simple_tokens=problem.startswith("aspect"))
            vectorizer.fit(list(zip(*data))[0])
            vectorizer.save(path)
        chunk_size = 2048
        batches = chunk_items(data, chunk_size)
        total = (len(data) + chunk_size - 1) // chunk_size

        for X, outputs in tqdm(batches, total=total, ascii="░▒▓▅", ncols=80):
            _texts = vectorizer.transform(X)
            for text, output in zip(_texts, outputs):
                if np.ndim(output) < 1:
                    output = np.expand_dims(output, axis=-1)
                yield text, output


def build_tfrecord(problem, part, features):
    tfrecord_path = os.path.join(
        DEFAULT_OUTPUT_DIR, "tfrecords", f"{problem}_{features}_{part}.tfrecord"
    )
    data = read_data(problem, part)
    if os.path.isfile(tfrecord_path):
        print(f"{tfrecord_path} already built...")
        return len(data)
    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    gen = generate_examples(problem, part, features)

    with tf.io.TFRecordWriter(tfrecord_path) as file_writer:
        for cnt, (text, output) in enumerate(gen):
            proto = {
                "text": tf.train.Feature(float_list=tf.train.FloatList(
                    value=np.array(text, dtype=np.float32).flatten().tolist()
                )),
                "output": tf.train.Feature(float_list=tf.train.FloatList(
                    value=np.array(output, dtype=np.float32).flatten().tolist()
                )),
            }
            if cnt < 8:
                print(np.shape(text), np.shape(output))
            record = tf.train.Example(
                features=tf.train.Features(feature=proto)
            ).SerializeToString()
            file_writer.write(record)
    return len(data)


def create_decode_fn(shape):

    @tf.function
    def decode_fn(example):
        data = tf.io.parse_single_example(example, {
            "text": tf.io.VarLenFeature(tf.float32),
            "output": tf.io.VarLenFeature(tf.float32),
        })
        return tf.reshape(data["text"].values, shape), data["output"].values

    return decode_fn


def tfrecord_build_load(problem, features, part, batch_size=256):
    num_steps = build_tfrecord(problem, part, features) // batch_size
    tfrecord_path = os.path.join(
        DEFAULT_OUTPUT_DIR, "tfrecords", f"{problem}_{features}_{part}.tfrecord"
    )
    num_features = reference[problem].get("num_features", 1)
    time_distributed = reference[problem].get("time_distributed", False)
    if features == "RoBERTa":
        padded_shapes = ([None, 768], [None])
        reshape = [-1, 768]
    else:
        padded_shapes= ([None], [None])
        reshape = [-1]

    return tf.data.TFRecordDataset([tfrecord_path]).\
        map(create_decode_fn(reshape), num_parallel_calls=12).\
        shuffle(batch_size * 8).repeat().\
        padded_batch(
            batch_size,
            padded_shapes=padded_shapes, #([None, dim], [None]),
            # padding_values=((float(RobertaTokenizer.from_pretrained("roberta-base").pad_token_id), 0.0),
            drop_remainder=False
        ).prefetch(48), num_steps


# {{{ keras utilities
class BCE(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def __call__(self, y_true, y_pred, **kwargs):
        return -tf.cast(y_true, tf.float32) * tf.math.log(y_pred + 1e-8) * (1 - self.weight) -\
            tf.cast(1 - y_true, tf.float32) * tf.math.log(1 - y_pred + 1e-8) * self.weight

    def get_config(self):
        return {"weight": self.weight}


class CCE(tf.keras.losses.Loss):
    def __init__(self, num_classes, weight=1.0, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.num_classes = num_classes
        self.axis = axis

    def __call__(self, y_true, y_pred, **kwargs):
        shape = np.ones(tf.keras.backend.ndim(y_pred), dtype=np.int32)
        shape[self.axis] = -1
        weights = [self.weight] + [(1 - self.weight) / (self.num_classes - 1) for _ in range(self.num_classes - 1)]
        weights = np.reshape(weights, shape)

        return -tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes) * tf.math.log(y_pred + 1e-8) * weights

    def get_config(self):
        return {"weight": self.weight, "axis": self.axis, "num_classes": self.num_classes}


class UAR(tf.keras.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recall_0 = tf.keras.metrics.Recall()
        self.recall_1 = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, **kwargs):
        self.recall_0.update_state(1 - y_true, 1 - y_pred)
        self.recall_1.update_state(y_true, y_pred)

    def reset_state(self):
        self.recall_0.reset_state()
        self.recall_1.reset_state()

    def result(self):
        return (self.recall_0.result() + self.recall_1.result()) * 0.5


class SparseUAR(tf.keras.metrics.Metric):
    def __init__(self, num_classes, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.axis = axis
        self.recalls = [tf.keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, **kwargs):
        for cls in range(self.num_classes):
            msk = tf.equal(tf.reshape(y_true, [-1]), cls)
            _ytrue = tf.reshape(tf.cast(msk, tf.int32), [-1])[msk]
            _ypred = tf.reshape(tf.cast(tf.equal(tf.argmax(y_pred, axis=self.axis), cls), tf.int32), [-1])[msk]
            tot = tf.reduce_sum(tf.cast(msk, tf.int32))
            if tot > 0:
                self.recalls[cls].update_state(_ytrue, _ypred)

    def reset_state(self):
        for cls in range(self.num_classes):
            self.recalls[cls].reset_state()

    def result(self):
        res = None
        for cls in range(self.num_classes):
            if res is None:
                res = self.recalls[cls].result()
            else:
                res = res + self.recalls[cls].result()
        return res / self.num_classes

    def get_config(self):
        dic = super().get_config()
        dic["num_classes"] = self.num_classes
        dic["axis"] = self.axis
        return dic
# }}}


def create_mlp_top_model(
    problem,
    features,
    lstm_layers=1,
    lstm_units=32,
    fc_layers=1,
    fc_units=128,
    embedding_dim=128,
    learning_rate=1e-3,
    loss_fn="binary_crossentropy",
    optimizer="Adam",
    final_activation="sigmoid",
    monitor_metric=None,
    num_features=1,
    weight=None,
    time_distributed=False,
    **_kwargs
):
    if features == "RoBERTa":
        out = inp = L.Input((None, 768), name="text")
    elif features == "BoW":
        out = inp = L.Input((None,), name="text")
        out = L.Embedding(2000, embedding_dim)(out)
    out = L.Dense(lstm_units, activation="relu")(out)
    for idx in range(lstm_layers):
        ret = time_distributed or idx + 1 < lstm_layers
        out = L.Bidirectional(L.LSTM(lstm_units, return_sequences=ret))(out)

    for idx in range(fc_layers):
        out = L.Dense(max(fc_units >> idx, 32), activation="relu")(out)
    out = L.Dense(
        num_features, activation=final_activation, name="output"
    )(out)
    name = f"MLP-{features}-{problem}-LSTM_{lstm_layers}_{lstm_units}-FC_{fc_layers}_{fc_units}-{optimizer}_%.8f" %\
        learning_rate
    metrics = []
    if monitor_metric == "uar" and loss_fn != "sparse_categorical_crossentropy":
        metrics.extend(["accuracy", UAR(name="uar")])
    elif monitor_metric == "uar" and loss_fn == "sparse_categorical_crossentropy":
        metrics.extend(["sparse_categorical_accuracy", SparseUAR(num_features, name="uar")])
    elif loss_fn == "binary_crossentropy":
        metrics.extend([monitor_metric, UAR(name="uar")])
    else:
        metrics.extend([monitor_metric])
    if weight is not None:
        if loss_fn == "binary_crossentropy":
            loss_fn = BCE(weight)
            metrics.append("binary_crossentropy")
            name += "_zw-%.8f" % weight
        elif loss_fn == "sparse_categorical_crossentropy":
            loss_fn = CCE(num_features, weight)
            metrics.append("sparse_categorical_crossentropy")
            name += "_zw-%.8f" % weight

    model = Model(inp, out, name=name)
    model.compile(
        getattr(optimizers, optimizer)(learning_rate), loss_fn, metrics=metrics
    )
    model.summary()
    return model


def train(problem, features, epochs=300, **kwargs):
    kwargs.update(reference[problem])
    print(kwargs)
    batch_size = kwargs.get("batch_size", 256)
    num_features = kwargs.get("num_features", 1)
    monitor_metric = "val_" + kwargs["monitor_metric"]
    mode = "min" if monitor_metric == "val_mean_absolute_error" else "max"
    baseline = kwargs.get("baseline", None)
    loss_fn = kwargs['loss_fn']

    train_data, train_steps = tfrecord_build_load(problem, features, "train", batch_size)
    val_data, valid_steps = tfrecord_build_load(problem, features, "valid", batch_size)
    model = create_mlp_top_model(problem, features, **kwargs)

    tbdir = os.path.join(DEFAULT_OUTPUT_DIR, "tb-graphs", model.name)
    modeldir = os.path.join(DEFAULT_OUTPUT_DIR, "models", model.name)
    csvsdir = os.path.join(DEFAULT_OUTPUT_DIR, "csvs")

    os.makedirs(tbdir, exist_ok=True)
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(csvsdir, exist_ok=True)

    cbs = [
        callbacks.TensorBoard(tbdir, write_graph=False),
        callbacks.CSVLogger(os.path.join(csvsdir, f"{model.name}.csv")),
        callbacks.ModelCheckpoint(
            modeldir, mode=mode, monitor=monitor_metric,
            verbose=2, save_best_only=True,
        ),
        callbacks.EarlyStopping(
            monitor=monitor_metric, patience=30, verbose=1, mode=mode,
            baseline=baseline,
        )
    ]
    hist = model.fit(
        train_data, validation_data=val_data, epochs=epochs, verbose=1,
        steps_per_epoch=train_steps, validation_steps=valid_steps, callbacks=cbs
    )
    if mode == "min":
        score = min(hist.history[monitor_metric])
    else:
        score = -max(hist.history[monitor_metric])
    return model, score


def make_train_function(problem, features, **kwargs):

    def train_fn(hparams={}):
        _, score = train(problem, features, **kwargs, **hparams)
        return score

    return train_fn


def hptune(problem, features, runs=25, **kwargs):
    configspace = ConfigurationSpace()
    configspace.add_hyperparameters([
        CH.UniformIntegerHyperparameter("fc_layers", 0, 1, default_value=1),
        CH.UniformIntegerHyperparameter(
            "fc_units", 32, 512, log=True, default_value=64
        ),
        CH.UniformIntegerHyperparameter(
            "lstm_units", 16, 64, log=True, default_value=32
        ),
        CH.UniformFloatHyperparameter(
            "learning_rate", 1e-6, 10, log=True, default_value=1e-3
        ),
    ])
    if reference[problem].get("use_weight", False):
        configspace.add_hyperparameter(
            CH.UniformFloatHyperparameter("weight", 1e-3, 1, log=True, default_value=1e-1)
        )

    os.makedirs(os.path.join(DEFAULT_OUTPUT_DIR, "hptune-outputs"), exist_ok=True)
    scenario = Scenario({
        "run_obj": "quality", "runcount-limit": runs, "cs": configspace,
        "output_dir": os.path.join(DEFAULT_OUTPUT_DIR, "hptune-outputs")
    })
    train_fn = make_train_function(problem, features, **kwargs)
    smac = FacadeOptimizer(scenario=scenario, tae_runner=train_fn)
    smac.optimize()

    return smac


def find_best_score(path, metric):
    monitor_metric = f"val_{metric}"
    mode = "min" if monitor_metric == "val_mean_absolute_error" else "max"
    df = pd.read_csv(os.path.join(DEFAULT_OUTPUT_DIR, "csvs", path))
    if monitor_metric not in df.columns:
        return None
        # monitor_metric = "val_accuracy"  # TODO: remove this hack
    df = df[monitor_metric]
    if mode == "max":
        return df.max().item(), path
    else:
        return -(df.min().item()), path


def find_best_model(problem, features, model="MLP"):
    scores = [
        find_best_score(p, reference[problem]["monitor_metric"])
        for p in os.listdir(os.path.join(DEFAULT_OUTPUT_DIR, "csvs"))
        if p.startswith(f"{model}-{features}-{problem}")
    ]
    scores = [score for score in scores if score is not None]
    return max(scores)[1][:-4]


def predict(problem, features, **kwargs):
    kwargs.update(reference[problem])
    data = list(generate_examples(problem, "test", features))
    inputs, outputs = zip(*data)
    model_path = find_best_model(problem, features)
    # _inputs = np.stack(inputs, axis=0)
    # _outputs = np.stack(outputs, axis=0)
    _inputs = [np.expand_dims(x, axis=0).tolist() for x in inputs]
    _outputs = [np.expand_dims(x, axis=0).tolist() for x in outputs]

    os.makedirs(os.path.join(DEFAULT_OUTPUT_DIR, "results"), exist_ok=True)
    print(kwargs, model_path)
    predmodel = tf.keras.models.load_model(
        os.path.join(DEFAULT_OUTPUT_DIR, "models", model_path),
        custom_objects={"UAR": UAR, "SparseUAR": SparseUAR, "BCE": BCE, "CCE": CCE},
    )
    predmodel.summary()

    predictions = [
        predmodel.predict(np.array(x))[0].tolist()
        for x in tqdm(_inputs, ascii="░▒▓▅", ncols=80)
    ]
    with open(os.path.join(DEFAULT_OUTPUT_DIR, "results", f"predictions_{problem}_{features}.json"), "w") as fl:
        json.dump(predictions, fl)
        print(f"wrote {fl.name}")
    with open(os.path.join(DEFAULT_OUTPUT_DIR, "results", f"inputs_{problem}_{features}.json"), "w") as fl:
        json.dump(_inputs, fl)
        print(f"wrote {fl.name}")
    with open(os.path.join(DEFAULT_OUTPUT_DIR, "results", f"outputs_{problem}_{features}.json"), "w") as fl:
        json.dump(_outputs, fl)
        print(f"wrote {fl.name}")
    obj = predmodel.evaluate(((np.array(x), np.array(y)) for x, y in zip(_inputs, _outputs)), batch_size=1, verbose=1)
    obj = dict(zip(predmodel.metrics_names, obj))

    with open(os.path.join(DEFAULT_OUTPUT_DIR, "results", f"metrics_{problem}_{features}.json"), "w") as fl:
        json.dump(obj, fl)
        print(f"wrote {fl.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "experiment",
        description="Script to train/predict results of an affective computing problems."
    )
    parser.add_argument("action", choices=["train", "hptune", "predict"])
    parser.add_argument("features", choices=["RoBERTa", "BoW"])
    parser.add_argument("problem", choices=list(reference.keys()))

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--fc-layers", type=int)
    parser.add_argument("--fc-units", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight", type=float)
    parser.add_argument("--runs", type=int)

    args = parser.parse_args()
    args = {k: v for k, v in args.__dict__.items() if v is not None}
    problem = args.pop("problem")
    features = args.pop("features")
    action_fn = globals()[args.pop("action")]

    action_fn(problem, features, **args)

