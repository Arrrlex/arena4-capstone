# %% [markdown]
# # Setup

# %% [markdown]
# ## Imports

# %%
import arena4_capstone.util as util

from arena4_capstone.models import gemma_2_2b_it, gemma_2_9b_it
from arena4_capstone.datasets.mcq import (
    create_mcq_dataset,
)
from arena4_capstone.datasets.tf_statements import (
    create_tf_statements_dataset_cot,
    create_tf_statements_dataset_simple,
)
from arena4_capstone.datasets.utils import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import torch as t
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import re
from typing import Callable, Optional
from functools import partial
import itertools


# %%
rng = np.random.RandomState(42)
sns.set_theme(style="whitegrid", context="paper")

# %% [markdown]
# ## Some common functions

# %%
def aggregate_judgements(judgements: pd.DataFrame):
    """
    Aggregate the judgements of a set of answers.

    Args:
        judgements: DataFrame with columns "Judgement" and "Prompt"

    Returns:
        DataFrame with columns "Judgement", "Prompt" and "Percentage", where Percentage
        represents what proportion of answers for that Prompt had that Judgement.
    """
    return (
        judgements.groupby("Prompt")["Judgement"]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index(name="Percentage")
    )


def judge_simple(row) -> str:
    if row.answer == row.correct_output:
        return "correct"
    elif row.answer == row.incorrect_output:
        return "incorrect"
    else:
        return "unclear"


def judge_with_answer_tags(row):
    try:
        match = re.search(r"<answer>(.*)</answer>", row.answer)
        answer = match.group(1)
    except AttributeError:
        return "unclear"

    if answer == row.correct_output:
        return "correct"
    elif answer == row.incorrect_output:
        return "incorrect"
    else:
        return "unclear"


def get_all_judgements(
    dataset: pd.DataFrame,
    judge: Callable,
    call_model: Callable,
    intervention: Optional[util.Intervention] = None,
):
    """
    Get the judgements for a dataset, both with and without an intervention.

    Returns a dataframe with columns "Judgement" and "Percentage".

    `judge` should take a row of the dataset, and return a judgement.

    `call_model` should take a pd.Series of prompts and an optional intervention, and return an answer.
    """

    print("Getting default judgements")
    judgements_default = dataset.assign(
        answer=lambda df: call_model(df.default_prompt),
        Judgement=lambda df: df.apply(judge, axis=1),
        Prompt="Default",
    )

    print("Getting lying judgements")
    judgements_lying = dataset.assign(
        answer=lambda df: call_model(df.lying_prompt),
        Judgement=lambda df: df.apply(judge, axis=1),
        Prompt="Lying",
    )

    if intervention is not None:
        print("Getting intervened judgements")
        judgements_intervened = dataset.assign(
            answer=lambda df: call_model(df.default_prompt, intervention=intervention),
            Judgement=lambda df: df.apply(judge, axis=1),
            Prompt="Default w/ Intervention",
        )
    else:
        judgements_intervened = None

    return pd.concat([judgements_default, judgements_lying, judgements_intervened])


# %%
next_token = util.vectorize(util.next_token_str, out_type="series")
next_logits = util.vectorize(util.next_logits, out_type="tensor")
continue_text = util.vectorize(util.continue_text, out_type="series", pbar=True)


# %% [markdown]
# ## Data 
# 
# We have 2 datasets of multiple-choice questions, "easy" (should be doable by 5-year-olds) and "hard" (should be doable by 12-year-olds).
# 
# For each dataset, we prepare the data as follows:
# 
# 1. Given a correct and an incorrect answer, we randomly choose one as "first answer" and one as "second answer"
# 2. We construct 2 prompts: a "default prompt" where we ask the model to choose a multiple choice answer, and a 
#    "lying prompt" where we explicitly ask it to lie.
# 3. We split the dataset into 75% train and 25% test

# %%
easy_mcq = create_mcq_dataset("mcq_5_yo.json", rng=rng)
hard_mcq = create_mcq_dataset("mcq_12_yo.json", rng=rng)
easy_train, easy_test = train_test_split(easy_mcq, train_fraction=0.75)
hard_train, hard_test = train_test_split(hard_mcq, train_fraction=0.75)

# %% [markdown]
# # Can the model even lie?
# 
# Before we proceed with interventions, we must first demonstrate that our chosen model (Gemma-2-2b-instruct) is capable of lying, and that our prompt elicits this behaviour successfully.
# 
# We demonstate this by running our 2 prompts through the model and checking whether it tells the truth (given the default prompt) and lies (given the lying prompt).

# %%
easy_judgements = get_all_judgements(
    dataset=easy_mcq,
    judge=judge_simple,
    call_model=partial(next_token, model=gemma_2_2b_it),
)

sns.catplot(
    data=aggregate_judgements(easy_judgements),
    x="Judgement",
    y="Percentage",
    col="Prompt",
    kind="bar",
    height=5,
    aspect=0.8,
)

plt.savefig(util.plots_dir / "mcq_easy_judgements.jpg")

# %% [markdown]
# As can be seen, the model is perfectly capable of lying, when prompted to do so.

# %% [markdown]
# # Extracting Lying Behaviour as Function Vector

# %%
interventions_train = util.ResidualStreamIntervention.batch_learn(
    model=gemma_2_2b_it,
    pos_prompts=hard_mcq.lying_prompt,
    neg_prompts=hard_mcq.default_prompt,
    layers=range(gemma_2_2b_it.config.num_hidden_layers),
    magnitudes=range(-3, 9),
)

# %% [markdown]
# `function_vecs` contains a vector for each hidden layer in our model. In order to calculate which layer is the best to intervene on, we will calculate the mean difference between the incorrect token's logit and the correct token's logit.

# %% [markdown]
# # Evaluating Function Vector Performance

# %% [markdown]
# ## Do these Function Vectors even make sense?

# %% [markdown]
# One way to evaluate this question is to look at, for each layer, how easily separable the activations are between lying and honest.

# %%
get_activations = util.vectorize(util.last_token_residual_stream, out_type="tensor")

lying_vectors = get_activations(easy_mcq.lying_prompt, model=gemma_2_2b_it).cpu().numpy()
honest_vectors = get_activations(easy_mcq.default_prompt, model=gemma_2_2b_it).cpu().numpy()


activations = np.vstack((lying_vectors, honest_vectors)).squeeze()

# Calculate PCA coordinates for each layer
n_layers = gemma_2_2b_it.config.num_hidden_layers
n_samples = len(easy_mcq)
pcas = [
    PCA(n_components=2).fit_transform(activations[:, layer, :])
    for layer in tqdm(range(n_layers), desc="Layers (for PCA)")
]

pca_coords = pd.DataFrame(
    [
        {
            "Prompt Type": prompt_type,
            "Sample": sample_idx,
            "Coord 1": layer_pca[offset + sample_idx, 0],
            "Coord 2": layer_pca[offset + sample_idx, 1],
            "Layer": layer,
        }
        for layer, layer_pca in enumerate(pcas)
        for prompt_type, offset in [("Lying", 0), ("Default", n_samples)]
        for sample_idx in range(n_samples)
    ]
)

# Create faceted scatter plot
g = sns.relplot(
    data=pca_coords,
    x="Coord 1",
    y="Coord 2",
    hue="Prompt Type",
    col="Layer",
    col_wrap=5,  # Number of columns
    alpha=0.7,
    kind="scatter",
    height=3,  # Height of each subplot
)

plt.savefig(util.plots_dir / "mcq_easy_pca.jpg")

# %% [markdown]
# ## Using Logit Difference

# %%
logit_diffs = t.zeros(gemma_2_2b_it.config.num_hidden_layers)

correct_token_ids = np.array(
    [
        gemma_2_2b_it.tokenizer.encode(choice, add_special_tokens=False)[0]
        for choice in easy_train.correct_output
    ]
)
incorrect_token_ids = np.array(
    [
        gemma_2_2b_it.tokenizer.encode(choice, add_special_tokens=False)[0]
        for choice in easy_train.incorrect_output
    ]
)

for layer in tqdm(range(gemma_2_2b_it.config.num_hidden_layers), desc="Layers"):
    intervention = interventions_train[layer, 1]

    logits = next_logits(
        easy_train.default_prompt,
        model=gemma_2_2b_it,
        intervention=intervention,
    )

    # Get the logits for the incorrect and correct answers
    incorrect_logits = logits[np.arange(logits.shape[0]), incorrect_token_ids]
    correct_logits = logits[np.arange(logits.shape[0]), correct_token_ids]

    # Calculate the logit difference
    logit_diffs[intervention.layer] = (incorrect_logits - correct_logits).mean()

# %%
df = pd.DataFrame(
    {"Layer": range(gemma_2_2b_it.config.num_hidden_layers), "Logit Difference": logit_diffs}
)
sns.lineplot(data=df, x="Layer", y="Logit Difference")
plt.title("Logit Difference by Layer")
plt.savefig(util.plots_dir / "mcq_easy_logit_diffs.jpg")


# %% [markdown]
# I have the feeling that this plot isn't telling us the whole story. Let's look a bit deeper, and investigate accuracies.

# %% [markdown]
# ## Using Accuracy across Layers and Magnitudes

# %%
lying_accuracies = []

for layer in tqdm(range(gemma_2_2b_it.config.num_hidden_layers), desc="Layers"):
    for coeff in tqdm(range(-3, 9), desc=f"Coeffs for layer {layer}"):
        judgements_intervened = easy_test.assign(
            answer=lambda df: next_token(
                df.default_prompt,
                model=gemma_2_2b_it,
                intervention=interventions_train[layer, coeff],
            ),
            Judgement=lambda df: df.apply(judge_simple, axis=1),
            Prompt="Default w/ Intervention",
        )
        intervention_stats = aggregate_judgements(judgements_intervened)
        lying_accuracies.append(intervention_stats.assign(coeff=coeff, layer=layer))

lying_accuracies = pd.concat(lying_accuracies, ignore_index=True)


# %%
def add_missing_judgement_rows(
    df: pd.DataFrame, judgements: tuple[str] = ("correct", "incorrect", "unclear")
) -> pd.DataFrame:
    """
    Add missing rows to ensure all combinations of (coef, layer, judgement) exist.
    """
    complete_df = df.copy()
    index = df.set_index(["coeff", "layer", "Judgement"]).index

    for coef, layer, judgement in itertools.product(df.coeff.unique(), df.layer.unique(), judgements):
        # Check if this combination exists in df
        if (coef, layer, judgement) not in index:
            # If not found, add a row with 0 values
            complete_df = util.append(
                complete_df,
                {
                    "coeff": coef,
                    "layer": layer,
                    "Judgement": judgement,
                    "Percentage": 0.0,
                },
            )

    return complete_df


# Update the lying_accuracies with complete data
lying_accuracies = add_missing_judgement_rows(lying_accuracies)

n_cols = 5
n_rows = (gemma_2_2b_it.config.num_hidden_layers + n_cols - 1) // n_cols

colours = {
    "correct": "#2E86C1",  # trustworthy blue
    "incorrect": "#E74C3C",  # warning red
    "unclear": "#95A5A6",  # neutral gray
}

g = sns.relplot(
    data=lying_accuracies,
    x="coeff",
    y="Percentage",
    hue="Judgement",
    col="layer",
    col_wrap=5,  # Number of columns
    palette=colours,
    kind="line",
    height=3,  # Height of each subplot
    alpha=0.7,
    linewidth=2.5,
    facet_kws={"sharey": True, "sharex": True}
)

# Customize the appearance
g.set_titles("Layer {col_name}", size=10)
g.set_axis_labels("Coefficient", "Percentage")

plt.savefig(util.plots_dir / "mcq_easy_lying_accuracies.jpg")


# %% [markdown]
# As you can see in the plot above, for early layers the intervention makes little difference. 
# 
# For layers 13-15, we see that the % of incorrect answers rises dramatically at a coefficient of between 1 and 2.5, but then if the coefficient continues to increase the incorrect answers are replaced with unclear answers (i.e. answers that are neither the correct nor the incorrect choice). It will be interesting to investigate these unclear answers further: are they garbage, or is the model answering the question, just not using the right format? If so, is it answering correctly or incorrectly?
# 
# For layers 16-20, the intervention never succeeds in getting the model to lie, but at a high enough coefficient the model outputs unclear answers.
# 
# Finally for layers 21-24, the intervention seems to shift incorrect and correct answers to a fixed number. My guess is that the model is simply giving the same answer for all questions. This would also explain why setting the coefficient negative gives us the reversed proportions of correct and incorrect.

# %% [markdown]
# # Going a little deeper

# %% [markdown]
# 
# 
# There are 3 things I want to investigate deeper:
# 
# 1. What's happening with large coefficients for layers 13, 14 and 15? Let's investigate these by manually inspecting a few of the answers given by the model
# 2. What's happening with large (and large negative) coefficients for the last few layers?
# 3. Does this lying vector generalise? Let's try a few different datasets:
#    1. What if instead of A and B, we use 1 and 2?
#    2. What if we try using the same function vector for true/false questions?
#    3. What about vice versa, i.e. learning a function vector for true/false questions using the same layer: does it work for true/false questions? Does it work for MCQ?
#    4. What if we try using a different dataset, i.e. the hard MCQ dataset? Can our model even reliably answer those questions?

# %%
interventions = util.ResidualStreamIntervention.batch_learn(
    model=gemma_2_2b_it,
    pos_prompts=easy_train.lying_prompt,
    neg_prompts=easy_train.default_prompt,
    layers=range(gemma_2_2b_it.config.num_hidden_layers),
    magnitudes=range(-3, 9),
)

# %% [markdown]
# ## Investigating middle layers and large coefficients

# %%
layer = 13
magnitude = 8
completions = util.batch_continue_text(
    easy_test.default_prompt, model=gemma_2_2b_it, intervention=interventions[layer, magnitude]
)
completions = pd.Series(completions, index=easy_test.index)

easy_test.assign(answer=completions)[["question", "choices_str", "answer"]]

# %% [markdown]
# It seems like the model is not lying when we crank up the coefficient on layer 13, but it answers either "..." or "that's a trick question!" for almost all questions.
# 
# "That's a trick question" is interesting, and it suggests perhaps the model thinks there is deception going on, even if it's not lying itself.

# %% [markdown]
# ## Investigating later layers and large coefficients

# %%
layer = 23
for coeff in [8, -3]:
    answers = next_token(
        easy_train.default_prompt, model=gemma_2_2b_it, intervention=interventions[layer, coeff]
    ).value_counts()
    print(f"Layer {layer}, Coeff {coeff}:")
    print(", ".join(f"{k}: {v}" for k, v in answers.items()))


# %% [markdown]
# Yep. As expected, at layer 24, pushing the vector just results in either A or B (more or less).
# 
# For some reason, at negative coefficients in the later layres, the model starts saying "Geplaatst" which is Dutch for "placed".

# %% [markdown]
# ## Investigating Generalization

# %% [markdown]
# ### Step 1: Different multiple-choice markers

# %%
layer = 13
magnitude = 2

easy_mcq_1_2 = create_mcq_dataset("mcq_5_yo.json", rng=rng, option_names=("1", "2"))

judgements_1_2 = get_all_judgements(
    easy_mcq_1_2,
    intervention=interventions[layer, magnitude],
    judge=judge_simple,
    call_model=partial(next_token, model=gemma_2_2b_it),
)


sns.catplot(
    data=aggregate_judgements(judgements_1_2),
    x="Judgement",
    y="Percentage",
    col="Prompt",
    kind="bar",
    height=5,
    aspect=0.8,
)

plt.savefig(util.plots_dir / "mcq_1_2_dataset.jpg")

# %%
lying_accuracies_1_2 = []

layer = 14

for coeff in tqdm(range(-3, 9)):
    judgements_intervened = easy_mcq_1_2.assign(
        answer=lambda df: next_token(
            df.default_prompt, model=gemma_2_2b_it, intervention=interventions[layer, coeff]
        ),
        Judgement=lambda df: df.apply(judge_simple, axis=1),
        Prompt="Default w/ Intervention",
    )
    intervention_stats = aggregate_judgements(judgements_intervened)
    lying_accuracies_1_2.append(
        intervention_stats.assign(coeff=coeff, layer=intervention.layer)
    )

lying_accuracies_df_1_2 = pd.concat(lying_accuracies_1_2, ignore_index=True)
lying_accuracies_df_1_2 = add_missing_judgement_rows(lying_accuracies_df_1_2)
# Create single plot
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=lying_accuracies_df_1_2,
    x="coeff",
    y="Percentage",
    hue="Judgement",
    palette=colours,
    markers=False,
)

plt.title(f"1 & 2 Dataset, Layer {layer}", fontsize=12)
plt.xlabel("Coefficient")
plt.ylabel("Percentage")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.savefig(util.plots_dir / "mcq_1_2_lying_accuracies.jpg")

# %% [markdown]
# It works! It generalises, at least somewhat, to 1 & 2.

# %% [markdown]
# ### Step 2: True-False Dataset

# %%
tf_dataset_cot = create_tf_statements_dataset_cot("tf_statements.json")
tf_dataset_simple = create_tf_statements_dataset_simple("tf_statements.json")

# %%
tf_cot_sample = tf_dataset_cot.sample(30)

# %%
layer = 13
magnitude = 2
judgements_tf_cot = get_all_judgements(
    tf_cot_sample,
    intervention=interventions[layer, magnitude],
    judge=judge_with_answer_tags,
    call_model=partial(
        continue_text, model=gemma_2_2b_it, max_new_tokens=200, intervention_pos="all_tokens"
    ),
)

sns.catplot(
    data=aggregate_judgements(judgements_tf_cot),
    x="Judgement",
    y="Percentage",
    col="Prompt",
    kind="bar",
    height=5,
    aspect=0.8,
)

plt.savefig(util.plots_dir / "tf_cot_sample.jpg")

# %%
layer = 13
magnitude = 2

judgements_tf_simple = get_all_judgements(
    tf_dataset_simple,
    intervention=interventions[(layer, magnitude)],
    judge=judge_simple,
    call_model=partial(next_token, model=gemma_2_2b_it),
)

sns.catplot(
    data=aggregate_judgements(judgements_tf_simple),
    x="Judgement",
    y="Percentage",
    col="Prompt",
    kind="bar",
    height=5,
    aspect=0.8,
)

plt.savefig(util.plots_dir / "tf_simple.jpg")


