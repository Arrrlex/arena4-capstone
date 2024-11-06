import json

import pandas as pd

import arena4_capstone.util as util
def train_test_split(df, train_fraction=0.75):
    train_size = int(len(df) * train_fraction)
    train_set = df.iloc[:train_size].reset_index(drop=True)
    test_set = df.iloc[train_size:].reset_index(drop=True)

    return train_set, test_set

def load_df(filename, shuffle: bool = True, random_state: int = 42):
    """
    Load a dataset from a json file.

    Args:
        filename (str): The name of the file to load.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        random_state (int, optional): The random state to use for shuffling. Defaults to 42.
    """
    with open(util.project_root / "datasets" / filename) as f:
        data = json.load(f)["data"]
    result_df = pd.DataFrame(data)
    if shuffle:
        result_df = result_df.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )
    return result_df

def turn(*lines: str, role: str = "user") -> str:
    return "\n".join([f"<start_of_turn>{role}", *lines, "<end_of_turn>"])

def user(*lines: str) -> str:
    return turn(*lines, role="user")

def model(*lines: str) -> str:
    return turn(*lines, role="model")

def combine(*prompts: str) -> str:
    return "\n".join(prompts) + "\n<start_of_turn>model\n"