import os
from multiprocessing.pool import ThreadPool as Pool

# import openai
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from tqdm import tqdm


# LLM = "gpt-3.5-turbo-0301"
LLM = "gpt-4-0314"

def create_chatgpt_func(system_prompt):
    def func(msg):
        llm = ChatOpenAI(model=LLM, max_retries=16)
        try:
            output = llm._generate(messages=[SystemMessage(content=system_prompt), HumanMessage(content=msg)])
            return output.generations[0].message.content
        except:
            return None
    return func


def create_gpt_compare_func(system_prompt):
    def func(msg):
        A, B = msg
        llm = ChatOpenAI(model=LLM, max_retries=16)
        output = llm._generate(messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=f'A: "{A}"\nB: "{B}"')
        ])
        return output.generations[0].message.content
    return func


def multi_map_reduce(func, df, col, update_path=None, start=0, end=None):
    responses = []
    end = end or len(df)
    iterable = np.copy(df[col].values[start:end])
    with Pool(8) as pool:
        for res in tqdm(pool.imap(func, iterable), total=len(iterable)):
            responses.append(res)
            if update_path is not None:
                sub_df = df[start:start+len(responses)].copy()
                sub_df["ChatGPT"] = responses
                sub_df.to_csv(os.path.join('chatgpt-responses', update_path))
    pool.join()
    return responses

# def create_callback(df, col, update_path):
#     def cb_func(responses):
#         sub_df = df[:len(responses)]
#         sub_df.loc[:len(responses) - 1, col] = responses
#         sub_df.to_csv(update_path)
#         return sub_df
#     return cb_func
