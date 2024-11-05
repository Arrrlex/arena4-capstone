from arena4_capstone.util import device, settings

import torch as t
from nnsight import LanguageModel
# from transformers import AutoTokenizer

import huggingface_hub as hf

hf.login(token=settings.HF_API_TOKEN)


if device.type == "cuda":
    t.cuda.empty_cache()

gemma = LanguageModel(
    "google/gemma-2-2b-it", device_map=device, token=settings.HF_API_TOKEN
)
