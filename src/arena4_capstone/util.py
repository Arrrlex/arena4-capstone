# %%
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Optional
import pandas as pd
import torch as t
from pydantic_settings import BaseSettings
import nnsight
from tqdm.auto import tqdm
from typing import cast

from openai import OpenAI

project_root = Path(__file__).parents[2]
plots_dir = project_root / "plots"

tqdm.pandas()

# %%


class Settings(BaseSettings):
    """
    Settings for the project.

    NNSIGHT_API_TOKEN is only required if REMOTE_MODE is True. This is obtained
      from https://login.ndif.us/
    """

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


def vectorize(func, *, out_type="list", threaded=False, pbar=False):
    def wrapper(first_arg, *args, **kwargs):
        first_arg = pd.Series(first_arg)

        # Function to be executed in parallel
        apply_func = functools.partial(func, *args, **kwargs)

        # Use ThreadPoolExecutor to map the function in parallel
        if threaded:
            with ThreadPoolExecutor() as executor:
                if pbar:
                    results_list = list(
                        tqdm(executor.map(apply_func, first_arg), total=len(first_arg))
                    )
                else:
                    results_list = list(executor.map(apply_func, first_arg))
        else:
            if pbar:
                results_list = list(first_arg.progress_map(apply_func))
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


# ===
# Interventions
# ===


# @dataclass
# class Intervention:
#     magnitude: float

#     @classmethod
#     def batch_learn(cls, model, pos_prompts, neg_prompts, magnitudes, **kwargs): ...

#     @classmethod
#     def learn(cls, model, pos_prompts, neg_prompts, magnitude=1.0, **kwargs): ...

#     def apply(self, model, prompt): ...


# @dataclass
# class ResidualStreamIntervention(Intervention):
#     layer: int
#     magnitude: float
#     vector: t.Tensor

#     def with_magnitude(self, magnitude):
#         return ResidualStreamIntervention(
#             layer=self.layer, magnitude=magnitude, vector=self.vector
#         )

#     @classmethod
#     def batch_learn(cls, model, pos_prompts, neg_prompts, layers, magnitudes):
#         get_residuals = vectorize(last_token_residual_stream, out_type="tensor")
#         pos_vectors = get_residuals(pos_prompts, model=model).mean(0)
#         neg_vectors = get_residuals(neg_prompts, model=model).mean(0)
#         function_vecs = pos_vectors - neg_vectors

#         return {
#             (layer, magnitude): cls(
#                 layer=layer, vector=function_vecs[layer], magnitude=magnitude
#             )
#             for layer in layers
#             for magnitude in magnitudes
#         }

#     @classmethod
#     def learn(cls, model, pos_prompts, neg_prompts, layer, magnitude=1.0):
#         interventions = cls.batch_learn(
#             model, pos_prompts, neg_prompts, [layer], [magnitude]
#         )
#         return interventions[(layer, magnitude)]

#     def apply(self, model):
#         model.model.layers[self.layer].output[0][:, -1, :] += (
#             self.vector * self.magnitude
#         )


class IntervenableModel:
    model: nnsight.LanguageModel
    train_dataset: pd.DataFrame
    interventions: list[t.Tensor]

    def __init__(self, model: nnsight.LanguageModel, train_dataset: pd.DataFrame):
        self.model = model
        self.train_dataset = train_dataset
        self.interventions = []

        pos_prompts = self.train_dataset.lying_prompt
        neg_prompts = self.train_dataset.default_prompt
        
        # Get residual activations
        get_residuals = vectorize(last_token_residual_stream, out_type="tensor")
        pos_vectors = get_residuals(pos_prompts, model=self.model).mean(0)
        neg_vectors = get_residuals(neg_prompts, model=self.model).mean(0)
        
        # Calculate intervention vectors (one tensor per layer)
        self.interventions: list[t.Tensor] = pos_vectors - neg_vectors


    # def trace(self, prompt: str, intervention: None | tuple[int, float] = None):
    #     # Create the trace context manager
    #     trace_ctx = self.model.trace(prompt, remote=settings.REMOTE_MODE)
        
    #     # Enter the context
    #     trace_ctx.__enter__()
        
    #     # Apply intervention if specified
    #     if intervention is not None:
    #         layer, coeff = intervention
    #         # Create and apply a temporary intervention
    #         self.model.model.layers[layer].output[0][:, -1, :] += (
    #             self.interventions[layer] * coeff
    #         )
        
    #     # Return the context manager to the caller
    #     return trace_ctx


@dataclass
class ModelWithIntervention:
    intervenable_model: IntervenableModel
    intervention: None | tuple[int, float]

    def apply(self) -> None:
        if self.intervention is not None:
            layer, coeff = self.intervention
            self.intervenable_model.model.layers[layer].output[0][:, -1, :] += (
                self.intervenable_model.interventions[layer] * coeff
            )
        


@t.inference_mode()
def next_logits(prompt: str, *, model: ModelWithIntervention):
    with model.intervenable_model.model.trace(prompt, remote=settings.REMOTE_MODE):
        if model.intervention is not None:
            model.apply()
        log_probs = model.intervenable_model.model.lm_head.output[..., -1, :].save()

    return log_probs.value.squeeze()



@t.inference_mode()
def next_token_str(prompt: str, *, model: ModelWithIntervention):

    logits = next_logits(prompt, model=model)

    return model.intervenable_model.model.tokenizer.decode(logits.argmax(), skip_special_tokens=False)


@t.inference_mode()
def last_token_residual_stream(
    prompt: str, *, model: ModelWithIntervention
):
    saves = []
    with model.intervenable_model.model.trace(prompt, remote=settings.REMOTE_MODE):
        model.apply()
        for _, layer in enumerate(model.intervenable_model.model.layers):
            saves.append(layer.output[0][:, -1, :].save())

    return t.stack([save.value for save in saves])

from typing import Literal
@t.inference_mode()
def continue_text(
    prompt: str,
    *,
    model: ModelWithIntervention,
    intervention_pos: Literal["last_input_token", "all_tokens"] = "last_input_token",
    max_new_tokens=50,
    skip_special_tokens=True,
):
    if intervention_pos not in ["last_input_token", "all_tokens"]:
        raise ValueError(f"Invalid intervention_pos: {intervention_pos}")
    with model.intervenable_model.model.generate(
        max_new_tokens=max_new_tokens, remote=settings.REMOTE_MODE
    ) as generator:
        with generator.invoke(prompt):
            if intervention is not None:
                intervention.apply(model)
            for _ in range(max_new_tokens):
                model.next()
                if intervention is not None and intervention_pos == "all_tokens":
                    intervention.apply(model)
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
    *,
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


def append(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)






def get_logit_diffs(model: nnsight.LanguageModel, dataset: pd.DataFrame, trained_interventions: dict[tuple[int, int], Intervention], intervention_coeff: int) -> t.Tensor:
    n_layers = cast(int, model.config.num_hidden_layers)

    logit_diffs = t.zeros(n_layers)

    correct_token_ids = t.Tensor(
        [
            model.tokenizer.encode(choice, add_special_tokens=False)[0]
            for choice in dataset.correct_output
        ]
    )
    incorrect_token_ids = t.Tensor(
        [
            model.tokenizer.encode(choice, add_special_tokens=False)[0]
            for choice in dataset.incorrect_output
        ]
    )

    for layer in tqdm(range(n_layers), desc="Layers"):
        intervention = trained_interventions[layer, intervention_coeff]

        logits: t.Tensor = next_logits( #type: ignore
            dataset.default_prompt,
            model=model,
            intervention=intervention,
        )

        # Get the logits for the incorrect and correct answers
        incorrect_logits = logits[t.arange(logits.shape[0]), incorrect_token_ids]
        correct_logits = logits[t.arange(logits.shape[0]), correct_token_ids]

        # Calculate the logit difference
        logit_diffs[intervention.layer] = (incorrect_logits - correct_logits).mean()
    
    return logit_diffs
