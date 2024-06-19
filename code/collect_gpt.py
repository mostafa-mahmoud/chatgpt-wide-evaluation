#!/usr/bin/env python3
# coding: utf-8

import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.cgpt import *
from core.pair_comparisons import *
from core.utils import *

sns.set_theme()
pd.set_option('display.width', 80)
pd.set_option('display.max_colwidth', 100)
warnings.filterwarnings('ignore')
os.listdir("datasets")


aspect_targets_prompt = """You are an aspect-based sentiment analysis expert,
you will be given a sentence by the user and you will list all the aspect words target objects.
List the words in bullet points.
The aspect targets are objects that are classified by a corresponding one of four sentiment targets: positive, negative, neutral, and conflict.
It is possible that a word has no target, which is defined as a background target. 
Use the following format:
* You will output a list of words in bullet points.
* Each bullet point will be on the form: "word" is target.
* The target is one of the four targets, do not report background targets.
* You will not mention any other text like "My guess is ..." or "I think ...".
* If all words have background target, then you return the word "BACKGROUND" without any bullet points.
"""

aspect_sentiment_prompt = """You are an aspect-based sentiment analysis expert,
you will be given a sentence by the user that contains aspect words objects.
Your task is to list all the sentiment opinionated words/expressions, that are corresponding to the aspect in the text (if any).
You just need to list the words/expression in bullet points without classifying them.
There will be many words without sentiment, these should not be listed.
Use the following format:
* You will output a list of words in bullet points.
* Each bullet point will be on the form (without quotations): "* expression"
* You should mention words that are explicitly in the text.
* You will not mention implied sentiment.
* You should mention the words exactly how they are written in the input, even if they have typos.
* You will not mention any other text like "My guess is ..." or "I think ...".
* If all words have no sentiment, then you respond with the word "BACKGROUND" without any bullet points.
"""

def run_aspect_extraction():
# {{{
    part = "test"
    for lap in ['res14', 'res15', 'lap14']:
        sentences = open(f"datasets/aspect extraction/data/{lap}/{part}/sentence.txt").read().split("\n")
        target_polarity = open(f"datasets/aspect extraction/data/{lap}/{part}/target_polarity.txt").read().split("\n")
        targets = open(f"datasets/aspect extraction/data/{lap}/{part}/target.txt").read().split("\n")
        opinions = open(f"datasets/aspect extraction/data/{lap}/{part}/opinion.txt").read().split("\n")

        # for sentence, labels, target, opinion in zip(sentences, target_polaritys, targets, opinions):
        df = pd.DataFrame({"text": sentences, "target_polarity": target_polarity, "target": targets, "opinion": opinions})

        multi_map_reduce(
            create_chatgpt_func(aspect_targets_prompt),
            df,
            'text',
            f'aspect_{lap}.csv',
            # end=20
        )
        multi_map_reduce(
            create_chatgpt_func(aspect_sentiment_prompt),
            df,
            'text',
            f'aspect_{lap}_opinion.csv',
            # end=20
        )
# }}}


intensity_prompt = """You are an expert at emotion analysis.
Given a pair of text A and B from the user,
you will output which text expresses higher intensity of the {emotion} emotion.
Use the following format:
* You are only allowed to answer "A" or "B".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write A or B, but nothing else.
"""


def run_emotion_intensity():
# {{{
    print_tree("datasets/intensity ranking")

    '''
    sadness_df = pd.read_csv("datasets/intensity ranking/emotion intensity/test/sadness.txt", delimiter="\t", names=["id", "text", "emotion", "score"])
    sadness_pairs = create_pairs_df(sadness_df, 'text', 'score')
    multi_map_reduce(
        create_gpt_compare_func(intensity_prompt.format(emotion="sadness")),
        sadness_pairs,
        ['text_a', 'text_b'],
        "sadness.csv",
    )

    joy_df = pd.read_csv("datasets/intensity ranking/emotion intensity/test/joy.txt", delimiter="\t", names=["id", "text", "emotion", "score"])
    joy_pairs = create_pairs_df(joy_df, 'text', 'score')

    multi_map_reduce(
        create_gpt_compare_func(intensity_prompt.format(emotion="joy")),
        joy_pairs,
        ['text_a', 'text_b'],
        'joy.csv',
    )
    '''

    fear_df = pd.read_csv("datasets/intensity ranking/emotion intensity/test/fear.txt", delimiter="\t", names=["id", "text", "emotion", "score"])
    fear_pairs = create_pairs_df(fear_df, 'text', 'score')

    multi_map_reduce(
        create_gpt_compare_func(intensity_prompt.format(emotion="fear")),
        fear_pairs,
        ['text_a', 'text_b'],
        'fear.csv',
    )

    anger_df = pd.read_csv("datasets/intensity ranking/emotion intensity/test/anger.txt", delimiter="\t", names=["id", "text", "emotion", "score"])
    anger_pairs = create_pairs_df(anger_df, 'text', 'score')
    multi_map_reduce(
        create_gpt_compare_func(intensity_prompt.format(emotion="anger")),
        anger_pairs,
        ['text_a', 'text_b'],
        'anger.csv',
    )
# }}}


sentiment_intensity_prompt = """You are an expert at sentiment analysis.
Given a pair of text A and B from the user,
you will output which text expresses more positive sentiment.
Use the following format:
* You are only allowed to answer "A" or "B".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write A or B, but nothing else.
"""


def run_sentiment_intensity():
# {{{
    #df = pd.read_json("datasets/intensity ranking/sentiment intensity/Microblog_Trainingdata.json")
    #df['spans'] = df['spans'].apply(lambda x: ' '.join(x))
    #df = df.rename(columns={"sentiment score": "sentiment"})
    #df = df.sample(frac=1., random_state=41)
    # df = filter_long_texts(df, 'spans')
    #df = df[1300:]
    df = pd.read_csv("datasets/intensity ranking/sentiment intensity/sentiment_intensity.csv")
    pairs = create_pairs_df(df, 'spans', 'sentiment')
    multi_map_reduce(
        create_gpt_compare_func(sentiment_intensity_prompt),
        pairs,
        ['spans_a', 'spans_b'],
        "sentiment_microblogs.csv",
    )
    # pd.read_json("datasets/intensity ranking/sentiment intensity/Headline_Trainingdata.json")
# }}}


personality_prompt = """You are an expert at the big-five personality traits assessment.
Given a pair of text A and B from the user,
you will output which text expresses higher intensity of the {trait} trait, from the big-five OCEAN personality traits.
Use the following format:
* You are only allowed to answer "A" or "B".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write A or B, but nothing else.
"""
def run_personality():
# {{{
    df = pd.read_csv("datasets/personality assessment/personality.csv")
    traits = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
    # traits = ['conscientiousness', 'openness']
    # traits = ['agreeableness', 'conscientiousness', 'openness']
    # traits = ["agreeableness"]
    print(traits)
    for trait in traits:
        pairs = create_pairs_df(df, 'text', trait)
        multi_map_reduce(
            create_gpt_compare_func(personality_prompt.format(trait=trait)),
            pairs,
            ['text_a', 'text_b'],
            f"personality_{trait}.csv",
        )
# }}}


well_being_prompt = """You are an expert at psyche analysis.
Given a text by the user, estimate if the given text talks about a stress-related topic, or expresses emotional stress be it implicit or explicit.
Use the following format:
* You are only allowed to answer "Yes" or "No".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write Yes or No, but nothing else.
"""

def run_well_being():
# {{{

    print_tree("datasets/well-being assessment")
    print(open("datasets/well-being assessment/data/README.md").read())

    well_being_twitter_full_df = pd.read_excel("datasets/well-being assessment/data/Twitter_Full.xlsx")
    well_being_twitter_full_df = well_being_twitter_full_df.sample(frac=1, random_state=41).reset_index()
    well_being_twitter_full_df = well_being_twitter_full_df[:1500]

    multi_map_reduce(
        create_chatgpt_func(well_being_prompt),
        well_being_twitter_full_df,
        'text',
        'well_being_twitter_full.csv',
    )

    well_being_twitter_df = pd.read_excel("datasets/well-being assessment/data/Twitter_Non-Advert.xlsx")
    well_being_twitter_df = well_being_twitter_df.sample(frac=1, random_state=41).reset_index()
    well_being_twitter_df = well_being_twitter_df[:1500]

    multi_map_reduce(
        create_chatgpt_func(well_being_prompt),
        well_being_twitter_df,
        'text',
        'well_being_twitter.csv'
    )

    reddit_df = pd.read_excel("datasets/well-being assessment/data/Reddit_Combi.xlsx")
    reddit_df = reddit_df.sample(frac=1, random_state=41).reset_index()
    reddit_df = reddit_df[:1500]
    multi_map_reduce(
        create_chatgpt_func(well_being_prompt),
        reddit_df,
        'title',
        'well_being_reddit.csv'
    )

    multi_map_reduce(
        create_chatgpt_func(well_being_prompt),
        reddit_df,
        'body',
        'well_being_reddit_body.csv'
    )

    reddit_titles_df = pd.read_excel("datasets/well-being assessment/data/Reddit_Title.xlsx")
    reddit_titles_df = reddit_titles_df.sample(frac=1, random_state=41).reset_index()
    reddit_titles_df = reddit_titles_df[:5000]
    multi_map_reduce(
        create_chatgpt_func(well_being_prompt),
        reddit_titles_df,
        'title',
        'well_being_reddit_titles.csv',
    )
# }}}


'''
def run_toxicity():
# {{{
    # TODO:


    print_tree("datasets/toxicity detection/data")

    pd.read_csv("datasets/toxicity detection/data/test.csv")
    pd.read_csv("datasets/toxicity detection/data/test_labels.csv")
    df = pd.read_csv("datasets/toxicity detection/data/test_labels.csv")
    pd.concat([df[c].value_counts() for c in df.columns[1:]], axis=1)
    pd.read_csv("datasets/toxicity detection/data/sample_submission.csv")
    df = pd.read_csv("datasets/toxicity detection/data/sample_submission.csv")
    pd.concat([df[c].value_counts() for c in df.columns[1:]], axis=1)
    pd.read_csv("datasets/toxicity detection/data/train.csv")
    df = pd.read_csv("datasets/toxicity detection/data/train.csv")
    pd.concat([df[c].value_counts() for c in df.columns[2:]], axis=1)
# }}}
'''


# engagement_prompt = """You are an expert at social media analysis.
# Given a text by the user, estimate if the given text is engaging or not.
# You will do this by estimating one of three labels, "1", "2", or "3".
# "1" denotes 0-9 retweets, "2" denotes 11-99 retweets", "3" denotes 100+ retweets.
# Use the following format:
# * You are only allowed to answer "1", "2" or "3".
# * Don't write an explanation of the answer.
# * Don't write things like "My guess is...", or "I think ...". Just write 1, 2, or 3, but nothing else.
# """
# 
# 
# def run_engagement():
# # {{{
#     os.listdir("datasets/engagement measurement")
#     print(open("datasets/engagement measurement/link.txt").read())
#     engagement_df = pd.read_csv("datasets/engagement measurement/TEDTalks.csv")
#     engagement_df = engagement_df.sample(frac=1, random_state=41).reset_index()
#     engagement_df = engagement_df[:4000]
#     # sns.histplot(np.log(engagement_df["retweetCount"])/np.log(10));
# 
#     multi_map_reduce(
#         create_chatgpt_func(engagement_prompt),
#         engagement_df,
#         'content',
#         'engagement.csv',
#     )
# }}}


engagement_prompt = """You are an expert at social media analysis.
Given a pair of texts A and B representing tweets, estimate which text is engaging more engaging.
You will achieve this by estimating which text is more viral,
by estimating which one has a higher number of retweets.
Use the following format:
* You are only allowed to answer "A" or "B".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write A or B, but nothing else.
"""


def run_engagement():
# {{{
    # os.listdir("datasets/engagement measurement")
    # print(open("datasets/engagement measurement/link.txt").read())
    engagement_df = pd.read_csv("datasets/engagement measurement/TEDTalks.csv")
    engagement_df = engagement_df.sample(frac=1, random_state=41).reset_index()
    # sns.histplot(np.log(engagement_df["retweetCount"])/np.log(10));
    engagement_df = engagement_df[:4000]

    eng_pairs = create_pairs_df(engagement_df, "content", "retweetCount")

    multi_map_reduce(
        create_gpt_compare_func(engagement_prompt),
        eng_pairs,
        ['content_a', 'content_b'],
        'engagement_pairs.csv',
        #'engagement_pairs_10600.csv',
        #start=10590
    )
# }}}


def run_subjectivity():
# {{{
    # TODO

    print_tree("datasets/subjectivity detection")
    print(len(open("datasets/subjectivity detection/rotten_imdb/quote.tok.gt9.5000", encoding="ISO-8859-1").readlines()))
    print(len(open("datasets/subjectivity detection/rotten_imdb/plot.tok.gt9.5000", encoding="ISO-8859-1").readlines()))
    # print(open("datasets/subjectivity detection/rotten_imdb/quote.tok.gt9.5000", encoding="ISO-8859-1").read())
    # print(open("datasets/subjectivity detection/rotten_imdb/plot.tok.gt9.5000").read())
    print(open("datasets/subjectivity detection/rotten_imdb/subjdata.README.1.0").read())
# }}}


sentiment_analysis_prompt = """You are an expert at sentiment analysis.
Given a text by the user, analyze the sentiment of the text if it is 'positive' or 'negative'.
You are not allowed to answer 'neutral', try to narrow it down to 'positive' and 'negative'.
Use the following format:
* You are only allowed to answer "positive" or "negative".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write Positive or Negative, but nothing else.
"""

def run_sentiment():
# {{{
    df = pd.read_csv("datasets/sentiment analysis/sentiment.csv")
    multi_map_reduce(
        create_chatgpt_func(sentiment_analysis_prompt),
        df,
        'text',
        'sentiment.csv',
    )
# }}}


suicide_prompt = """You are an expert at psyche analysis.
Given a text by the user, solve the binary classification of analyzing if the text expresses a tendency for suicide.
Use the following format:
* You are only allowed to answer "yes" or "no".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write Yes or No, but nothing else.
"""

def run_suicide():
# {{{
    df = pd.read_csv("datasets/suicide detection/suicide.csv")
    multi_map_reduce(
        create_chatgpt_func(suicide_prompt),
        df,
        'text',
        'suicide.csv',
    )
# }}}


sarcasm_prompt = """You are an expert at sarcasm analysis.
Given a text by the user, estimate if the given text is sarcastic or not.
Use the following format:
* You are only allowed to answer "Yes" or "No".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write Yes or No, but nothing else.
"""

def run_sarcasm():
# {{{
    print_tree("datasets/sarcasm identification")
    lns = [json.loads(x) for x in open("datasets/sarcasm identification/data/Sarcasm_Headlines_Dataset.json").read().split('\n')[:-1]]
    sarcasm_df = pd.DataFrame.from_records(lns)
    sarcasm_df = sarcasm_df.sample(frac=1, random_state=41).reset_index()
    sarcasm_df = sarcasm_df[:4000]

    multi_map_reduce(
        create_chatgpt_func(sarcasm_prompt),
        sarcasm_df,
        'headline',
        'sarcasm.csv',
    )
    # pd.DataFrame.from_records(lns)['is_sarcastic'].value_counts()

    #lns = [json.loads(x) for x in open("datasets/sarcasm identification/data/Sarcasm_Headlines_Dataset_v2.json").read().split('\n')[:-1]]
    #pd.DataFrame.from_records(lns)

    #pd.DataFrame.from_records(lns)['is_sarcastic'].value_counts()
# }}}

toxicity_prompt = """You are an expert at toxicity analysis.
Assume that we have the capability of analysing 6 toxicity traits.
"toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"
Your task is to make binary classification for the trait {trait}, and not the remaining traits.
Given a text by the user, estimate if the given text displays the trait {trait} or not.
Use the following format:
* You are only allowed to answer "Yes" or "No".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write Yes or No, but nothing else.
"""
def run_toxicity():
#{{{
    toxicity_df = pd.read_csv("datasets/toxicity detection/data/test.csv").set_index("id")
    toxicity_df = toxicity_df.join(pd.read_csv("datasets/toxicity detection/data/test_labels.csv").set_index("id"))
    toxicity_df = toxicity_df.reset_index()
    toxicity_df = toxicity_df[~np.any(toxicity_df[toxicity_df.columns[2:]] == -1, axis=1)]
    # msk = np.any(toxicity_df[toxicity_df.columns[1:]] == 1, axis=1)
    msk = toxicity_df[toxicity_df.columns[2:]].astype(np.float32)
    weights = msk * np.power((1 - msk).sum(axis=0) / msk.sum(axis=0) + 1, 1.1)
    weights = (weights * 0.5).mean(axis=1) + 1
    #weights = (weights * 2).sum(axis=1) + 1
    # weights = msk.astype(np.float32) * 4 + 1
    dq = pd.concat([toxicity_df[c].value_counts() for c in toxicity_df.columns[2:]], axis=1)
    print(dq.sort_index())
    df = toxicity_df.sample(frac=1, random_state=41, weights=weights).reset_index()[:1000]
    dq = pd.concat([df[c].value_counts() for c in df.columns[3:]], axis=1)
    print(dq.sort_index())

    print(df.shape, df.columns[3:])
    # return
    for trait in df.columns[3:]:
        multi_map_reduce(
            create_chatgpt_func(toxicity_prompt.format(trait=trait)),
            df,
            'comment_text',
            'toxicity_{trait}.csv'.format(trait=trait),
        )
#}}}


subjectivity_prompt = """You are an expert at language and sentiment analysis.
The user will give you a text, your task is to make a binary classification on the text,
if the given text is opinionated/subjective/biased, or if it is non-opinionated/objective/descriptive/factual.
Please note that this is about "how" the text is described and not "what" it describes,
so the text can still "objectively" describe a fictional story with some emotional terms.
Use the following format:
* You are only allowed to answer "objective" or "subjective".
* Don't write an explanation of the answer.
* Don't write things like "My guess is...", or "I think ...". Just write one of the two labels, but nothing else.
"""

def run_subjectivity():
    subjective = open("datasets/subjectivity detection/rotten_imdb/quote.tok.gt9.5000", encoding="ISO-8859-1").read().split('\n')
    objective = open("datasets/subjectivity detection/rotten_imdb/plot.tok.gt9.5000", encoding="ISO-8859-1").read().split('\n')

    subjectivity_df = pd.concat([
        pd.DataFrame({"text": subjective, "label": 1}),
        pd.DataFrame({"text": objective, "label": 0}),
    ])
    subjectivity_df = subjectivity_df.sample(frac=1, random_state=41).reset_index()
    subjectivity_df = subjectivity_df[:2000]

    multi_map_reduce(
        create_chatgpt_func(subjectivity_prompt),
        subjectivity_df,
        'text',
        'subjectivity.csv',
        # end=20
    )


def print_prompts():
    dx = {k: v for k, v in globals().items()}
    for k, v in dx.items():
        if k.endswith("_prompt"):
            title = k.replace("_", " ").capitalize()
            prompt = v.replace("\n* ", "\\\\\n* ")
            # print(f"{k}\n\n```\n{v}\n```\n\n============\n\n")
            print("\\item %s\n\\begin{myquote}\n%s\end{myquote}\n\n" % (title, prompt))


if __name__ == "__main__":
    print_prompts()
    run_emotion_intensity()  # DONE
    print('\n=======================================================\n' * 3)
    run_engagement()  # DONE
    print('\n=======================================================\n' * 3)
    run_subjectivity()   # DONE
    print('\n=======================================================\n' * 3)
    run_sentiment_intensity()  # DONE
    print('\n=======================================================\n' * 3)
    run_toxicity()  # DONE
    print('\n=======================================================\n' * 3)
    run_well_being()  # DONE
    print('\n=======================================================\n' * 3)
    run_sentiment()
    print('\n=======================================================\n' * 3)
    run_aspect_extraction()
    print('\n=======================================================\n' * 3)
    run_suicide()  # DONE
    print('\n=======================================================\n' * 3)
    run_personality()
    print('\n=======================================================\n' * 3)
    run_sarcasm()  # DONE
    print('\n=======================================================\n' * 3)
