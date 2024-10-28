#%%
import einops
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from util import *

# Replace this with your actual activations and labels
# activations: numpy array of shape (n_samples, n_features)
# labels: numpy array of shape (n_samples,)
import itertools
from util import *


import os

#%%

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
!export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

rng = np.random.RandomState(42)


def mix_options(row: pd.Series):
    correct_answer = row["correct answer"]
    incorrect_answer = row["incorrect answer"]

    if rng.random() < 0.5:
        answer_a = correct_answer
        answer_b = incorrect_answer
        correct_answer_str = "A"
    else:
        answer_a = incorrect_answer
        answer_b = correct_answer
        correct_answer_str = "B"

    row["A"] = answer_a
    row["B"] = answer_b
    row["correct"] = correct_answer_str
    row["choices"] = f"A. {answer_a}\nB. {answer_b}"
    return row

task_lying_prefix = (
    "I'm playing a game: I'll give you a multiple choice question, and I'd like you to answer as if "
    "you were dishonest, i.e. deliberately give me the wrong answer."
)
task_honest_prefix = (
    "I'm playing a game: I'll give you a multiple choice question, and I'd like you to answer honestly, "
    "i.e. deliberately give me the right answer."
)
none_prefix = "I'll give you a multiple choice question, and I'd like you to answer A or B. Only output A or B, and nothing else."


def mcq_prompt(data, prompts: list[str]):
    return combine(*prompts, data.question, data.choices)


# Shuffle the mcq_easy DataFrame and split into train and test sets
def train_test_split(df, train_fraction=0.75, random_state=42):
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_size = int(len(df_shuffled) * train_fraction)
    train_set = df_shuffled.iloc[:train_size]
    test_set = df_shuffled.iloc[train_size:]
    return train_set, test_set

def prepare_mcq(df):
    # Mix options for each row
    df = df.apply(mix_options, axis=1)

    # Add prompts
    df["lying_prompt"] = mcq_prompt(df, [task_lying_prefix])
    df["honest_prompt"] = mcq_prompt(df, [task_honest_prefix])
    df["none_prompt"] = mcq_prompt(df, [none_prefix])

    # Split into train and test sets
    train_set, test_set = train_test_split(df)

    return train_set, test_set
#%%
# For demonstration purposes, let's create synthetic data
# Let's assume two behaviors/classes

# Generate synthetic activations for two classes
# np.random.seed(42)
# activations_class1 = np.random.normal(loc=0.0, scale=1.0, size=(n_samples // 2, n_features))
# activations_class2 = np.random.normal(loc=2.0, scale=1.0, size=(n_samples // 2, n_features))

# activations = np.vstack((activations_class1, activations_class2))
# labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

hard_train, hard_test = prepare_mcq(load_df("mcq_12_yo.json"))
hard_lying_vectors = last_token_residual_stream(hard_train.lying_prompt, gemma, as_tensor=True)
hard_honest_vectors = last_token_residual_stream(hard_train.honest_prompt, gemma, as_tensor=True)

#%%
hard_lying_vectors = einops.rearrange(hard_lying_vectors, "b l d_model -> l b d_model")
hard_honest_vectors = einops.rearrange(hard_honest_vectors, "b l d_model -> l b d_model")

print(hard_lying_vectors.shape)
print(hard_honest_vectors.shape)

#%%

lying_activations = hard_lying_vectors[25].cpu().numpy()
honest_activations = hard_honest_vectors[25].cpu().numpy()

#%%
# With 3 PCA components
activations = np.vstack((lying_activations, honest_activations))

labels = np.array([0] * (30) + [1] * (30))


# Perform PCA to reduce to 3 components
pca = PCA(n_components=3)
activations_pca = pca.fit_transform(activations)

# Plot the projected activations
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
scatter = ax.scatter(
    activations_pca[:, 0],
    activations_pca[:, 1],
    # activations_pca[:, 2],
    c=labels,
    cmap='viridis',
    alpha=0.7
)
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
plt.title('PCA Projection of Activations')
plt.show()


# %%
# With 2 PCA components
pca = PCA(n_components=2)
activations_pca = pca.fit_transform(activations)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    activations_pca[:, 0],
    activations_pca[:, 1],
    c=labels,
    cmap='viridis',
    alpha=0.7
)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection of Activations (2 Components)')
plt.show()
# %%
n_layers = gemma.model.layers
pca_per_layer = np.zeros((n_layers, 60))
for i in range(n_layers):
    pca = PCA(n_components=1)

    lying_activations = hard_lying_vectors[i].cpu().numpy()
    honest_activations = hard_honest_vectors[i].cpu().numpy()

    activations = np.vstack((lying_activations, honest_activations))

    print(activations.shape)

    pca_per_layer.append(pca.fit_transform(activations).flatten())

    # Plot the projected activations
    plt.figure(figsize=(12, 6))

    # Option 1: 1D Scatter Plot
    plt.scatter(pca_per_layer, np.arange(n_layers), c=labels, cmap='viridis', alpha=0.7, marker='|')
    plt.yticks(np.arange(n_layers))
    plt.xlabel('Principal Component 1')
    plt.title('PCA Projection of Activations (1 Component)')
    plt.legend(*plt.gca().get_legend_handles_labels(), title="Classes")
    plt.show()
# %%
