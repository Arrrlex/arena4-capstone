# %%
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import functools
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import torch as t
from pydantic_settings import BaseSettings
import nnsight

from openai import OpenAI

project_root = Path(__file__).parents[2]
plots_dir = project_root / "plots"

# %%


class Settings(BaseSettings):
    HF_API_TOKEN: str
    NNSIGHT_API_TOKEN: str = None
    OPENAI_API_TOKEN: str = None
    REMOTE_MODE: bool = False

    class Config:
        env_file = str(project_root / ".env")
        env_file_encoding = "utf-8"


settings = Settings()

device = t.device("cuda" if t.cuda.is_available() else "cpu")

if settings.REMOTE_MODE:
    nnsight.CONFIG.set_default_api_key(settings.NNSIGHT_API_TOKEN)

# %%


def load_df(filename, shuffle: bool = True, random_state: int = 42):
    """
    Load a dataset from a json file.

    Args:
        filename (str): The name of the file to load.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        random_state (int, optional): The random state to use for shuffling. Defaults to 42.
    """
    with open(project_root / "datasets" / filename) as f:
        data = json.load(f)["data"]
    result_df = pd.DataFrame(data)
    if shuffle:
        result_df = result_df.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
    return result_df


def map_with(f, a: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Map a function over a series and a dataframe row-wise.

    Returns a series `out` where `out[i] = f(a[i], df.iloc[i])` for all `i`.
    """
    assert a.index.equals(df.index)
    assert len(a) == len(df)
    return pd.Series([f(a_i, df.iloc[i]) for i, a_i in enumerate(a)], index=a.index)


def combine(*lines: str) -> str:
    """
    Combine a list of strings into a single string with a turn delimiter.
    """
    start = "<start_of_turn>user"
    middle = "\n".join(lines)
    end = "<end_of_turn>\n<start_of_turn>model\n"

    return "\n".join([start, middle, end])

def vectorize(func, out_type="list", threaded=False):
    def wrapper(first_arg, *args, **kwargs):
        first_arg = pd.Series(first_arg)

        # Function to be executed in parallel
        apply_func = functools.partial(func, *args, **kwargs)

        # Use ThreadPoolExecutor to map the function in parallel
        if threaded:
            with ThreadPoolExecutor() as executor:
                results_list = list(executor.map(apply_func, first_arg))
        else:
            results_list = list(first_arg.map(apply_func))

        if out_type == "tensor":
            results = t.stack(results_list, dim=0)
        elif out_type == "series":
            results = pd.Series(results_list, index=first_arg.index)
        elif out_type == "list":
            pass  # results is already a list
        else:
            raise ValueError(f"Invalid out_type: {out_type}")

        return results

    return wrapper


@dataclass
class Intervention:
    magnitude: float

    @classmethod
    def learn(cls, model, pos_prompts, neg_prompts, magnitude=1.0, **kwargs): ...

    def apply(self, model, prompt): ...


@dataclass
class ResidualStreamIntervention(Intervention):
    layer: int
    magnitude: float
    vector: t.Tensor

    def with_magnitude(self, magnitude):
        return ResidualStreamIntervention(
            layer=self.layer, magnitude=magnitude, vector=self.vector
        )

    @classmethod
    def batch_learn(cls, model, pos_prompts, neg_prompts, layers, magnitudes):
        get_residuals = vectorize(last_token_residual_stream, out_type="tensor")
        pos_vectors = get_residuals(pos_prompts, model=model).mean(0)
        neg_vectors = get_residuals(neg_prompts, model=model).mean(0)
        function_vecs = pos_vectors - neg_vectors

        return {
            (layer, magnitude): cls(
                layer=layer, vector=function_vecs[layer], magnitude=magnitude
            )
            for layer in layers
            for magnitude in magnitudes
        }

    @classmethod
    def learn(cls, model, pos_prompts, neg_prompts, layer, magnitude):
        return cls.batch_learn(model, pos_prompts, neg_prompts, [layer], [magnitude])[0]

    def apply(self, model):
        model.model.layers[self.layer].output[0][:, -1, :] += self.vector


@t.inference_mode()
def next_logits(prompt: str, model, intervention: Optional[Intervention] = None):
    with model.trace(prompt, remote=settings.REMOTE_MODE):
        if intervention is not None:
            intervention.apply(model)
        log_probs = model.lm_head.output[..., -1, :].save()

    return log_probs.value.squeeze()


@t.inference_mode()
def next_token_str(prompt: str, model, intervention: Optional[Intervention] = None):
    logits = next_logits(prompt, model, intervention)

    return model.tokenizer.decode(logits.argmax(), skip_special_tokens=False)


@t.inference_mode()
def last_token_residual_stream(
    prompt: str, model, intervention: Optional[Intervention] = None
):
    saves = []
    with model.trace(prompt, remote=settings.REMOTE_MODE):
        if intervention is not None:
            intervention.apply(model)
        for _, layer in enumerate(model.model.layers):
            saves.append(layer.output[0][:, -1, :].save())

    return t.stack([save.value for save in saves])



def accuracy(answers, df, comp=lambda a, c: a == c.correct):
    judgements = pd.Series([comp(a, c) for a, (_, c) in zip(answers, df.iterrows())])

    return judgements.mean()


@t.inference_mode()
def continue_text(
    prompt: str,
    model,
    intervention: Optional[Intervention] = None,
    max_new_tokens=50,
    skip_special_tokens=True,
):
    with model.generate(
        max_new_tokens=max_new_tokens, remote=settings.REMOTE_MODE
    ) as generator:
        with generator.invoke(prompt):
            if intervention is not None:
                intervention.apply(model)
            for _ in range(max_new_tokens):
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


@t.inference_mode()
def batch_continue_text(
    prompts,
    model,
    intervention: Optional[Intervention] = None,
    max_new_tokens=50,
    skip_special_tokens=True,
):
    with model.generate(
        max_new_tokens=max_new_tokens, remote=settings.REMOTE_MODE
    ) as generator:
        with generator.invoke(list(prompts)):
            if intervention is not None:
                intervention.apply(model)
            for _ in range(max_new_tokens):
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


openai_client = OpenAI(api_key=settings.OPENAI_API_TOKEN)


def call_openai(prompt, return_type, model="gpt-4o-2024-08-06"):
    completion = openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=return_type,
    )

    value = completion.choices[0].message.parsed
    return value


def train_test_split(df, train_fraction=0.75):
    train_size = int(len(df) * train_fraction)
    train_set = df.iloc[:train_size].reset_index(drop=True)
    test_set = df.iloc[train_size:].reset_index(drop=True)

    return train_set, test_set


def append(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
