# %%
import random
from typing import Literal
from textwrap import dedent
import json
from pathlib import Path
import pandas as pd
import gc
import json
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import functools
import pandas as pd
import dotenv
from arena.plotly_utils import imshow

from pydantic import BaseModel, Field
from openai import OpenAI
import functools
from concurrent.futures import ThreadPoolExecutor

project_root = Path(__file__).parent.parent.parent



# %%
API_TOKEN = open(project_root / "token.txt").read()

t.cuda.empty_cache()

device = t.device("cuda" if t.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it", use_auth_token=API_TOKEN
)

gemma = LanguageModel("google/gemma-2-9b-it", device_map=device, token=API_TOKEN)

# %%


project_root = Path(__file__).parent.parent.parent

dotenv.load_dotenv(project_root / ".." / ".env")

def load_df(filename, shuffle = True, random_state=42):
    with open(project_root / "datasets" / filename) as f:
        data = json.load(f)["data"]
    result_df = pd.DataFrame(data)
    if shuffle:
        result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result_df



def map_with(f, a: pd.Series, df: pd.DataFrame):
    assert a.index.equals(df.index)
    assert len(a) == len(df)
    return pd.Series([f(a_i, df.iloc[i]) for i, a_i in enumerate(a)], index=a.index)


def combine(*l):
    sot = "<start_of_turn>"
    eot = "<end_of_turn>"

    ans = f"{sot}user\n"
    for elem in l:
        ans = ans + elem + "\n"

    ans = ans + f"{eot}\n{sot}model\n"

    return ans






def vectorizable(func):
    @functools.wraps(func)
    def wrapper(first_arg, *args, **kwargs):
        if isinstance(first_arg, pd.Series) or isinstance(first_arg, list):
            first_arg = pd.Series(first_arg)

            as_tensor = kwargs.pop("as_tensor", False)
            threaded = kwargs.pop("threaded", False)
            
            # Function to be executed in parallel
            def apply_func(x):
                return func(x, *args, **kwargs)

            # Use ThreadPoolExecutor to map the function in parallel
            if threaded:
                with ThreadPoolExecutor() as executor:
                    first_arg.iloc[:] = list(executor.map(apply_func, first_arg))
                    results = first_arg

            else:
                results = first_arg.map(apply_func)
            
            if as_tensor:
                return t.stack(list(results), dim=0)
            else:
                return results

        else:
            return func(first_arg, *args, **kwargs)

    return wrapper


@vectorizable
@t.inference_mode()
def next_logits(prompt: str, model, intervention: None | tuple[int, t.Tensor] = None):
    with model.trace(prompt) as tracer:
        if intervention is not None:
            layer, steering = intervention
            model.model.layers[layer].output[0][:, -1, :] += steering
        log_probs = model.lm_head.output[..., -1, :].save()

    log_probs = log_probs.value

    assert log_probs.shape == (1, model.config.vocab_size)

    return log_probs.squeeze()


@vectorizable
def next_token_str(
    prompt: str, model, intervention: None | tuple[int, t.Tensor] = None
):
    logits = next_logits(prompt, model, intervention)

    assert logits.shape == (model.config.vocab_size,)
    return model.tokenizer.decode(logits.argmax(), skip_special_tokens=False)


@vectorizable
@t.inference_mode()
def last_token_residual_stream(prompt: str, model):
    saves = []
    with model.trace(prompt):
        for _, layer in enumerate(model.model.layers):
            saves.append(layer.output[0][:, -1, :].save())

    saves = [save.value for save in saves]

    tensor = t.stack(saves).squeeze()

    assert tensor.shape == (model.config.num_hidden_layers, model.config.hidden_size)
    return tensor


def last_token_batch_mean(prompts: pd.Series, model):
    residuals = last_token_residual_stream(prompts, model)

    residuals = t.stack(list(residuals), dim=0)

    assert residuals.shape == (
        len(prompts),
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )

    return residuals.mean(dim=0)


def accuracy(answers, df, comp=lambda a, c: a == c.correct):
    judgements = pd.Series([comp(a, c) for a, (_, c) in zip(answers, df.iterrows())])

    return judgements.mean()


@vectorizable
def continue_text(
    prompt: str,
    model,
    intervention: None | tuple[int, t.Tensor] = None,
    max_new_tokens=50,
    skip_special_tokens=True,
):
    with model.generate(max_new_tokens=50) as generator:
        with generator.invoke(prompt):
            if intervention is not None:
                layer, vector = intervention
                model.model.layers[layer].output[0][:, -1, :] += vector
            for n in range(50):
                model.next()
            all_tokens = model.generator.output.save()

    complete_string = model.tokenizer.batch_decode(
        all_tokens.value, skip_special_tokens=False
    )[0]
    # Find the first occurrence of the original prompt
    prompt_index = complete_string.find(prompt)
    assert prompt_index != -1, "Original prompt not found in the completion"

    # Ensure it's the only occurrence
    assert (
        complete_string.count(prompt) == 1
    ), "Multiple occurrences of the original prompt found"

    # Keep only the text coming after the prompt
    complete_string = complete_string[prompt_index + len(prompt) :]

    if skip_special_tokens:
        # Re-encode and decode the completion to remove special tokens
        tokens = model.tokenizer.encode(complete_string)
        complete_string = model.tokenizer.decode(tokens, skip_special_tokens=True)

    return complete_string


# def map_with(f, a:pd.Series, df: pd.DataFrame):


def batch_continue_text(
    prompts,
    model,
    intervention: None | tuple[int, t.Tensor] = None,
    max_new_tokens=50,
    skip_special_tokens=True,
):
    with model.generate(max_new_tokens=50) as generator:
        with generator.invoke(list(prompts)):
            if intervention is not None:
                layer, vector = intervention
                model.model.layers[layer].output[0][:, -1, :] += vector
            for n in range(50):
                model.next()
            all_tokens = model.generator.output.save()

    complete_strings = model.tokenizer.batch_decode(
        all_tokens.value, skip_special_tokens=False
    )

    processed_completions = []
    # Find the first occurrence of the original prompt
    for prompt, complete_string in zip(prompts, complete_strings):
        prompt_index = complete_string.find(prompt)
        assert prompt_index != -1, "Original prompt not found in the completion"

        # Ensure it's the only occurrence
        assert (
            complete_string.count(prompt) == 1
        ), "Multiple occurrences of the original prompt found"

        # Keep only the text coming after the prompt
        complete_string = complete_string[prompt_index + len(prompt) :]

        if skip_special_tokens:
            # Re-encode and decode the completion to remove special tokens
            tokens = model.tokenizer.encode(complete_string)
            complete_string = model.tokenizer.decode(tokens, skip_special_tokens=True)

        processed_completions.append(complete_string)

    return processed_completions

@vectorizable
def openai_api(prompt, return_type, model="gpt-4o-mini"):
    from pydantic import BaseModel
    from openai import OpenAI

    client = OpenAI()


    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=return_type,
    )

    value = completion.choices[0].message.parsed
    return value
