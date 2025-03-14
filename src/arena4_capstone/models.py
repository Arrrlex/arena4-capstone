from arena4_capstone.util import device, settings

import torch as t
from nnsight import LanguageModel
# from transformers import AutoTokenizer

import huggingface_hub as hf

hf.login(token=settings.HF_API_TOKEN)


if device.type == "cuda":
    t.cuda.empty_cache()

gemma_2_2b_it: LanguageModel = LanguageModel(
    "google/gemma-2-2b-it", device_map=device, token=settings.HF_API_TOKEN
)

gemma_2_9b_it: LanguageModel = LanguageModel(
    "google/gemma-2-9b-it", device_map=device, token=settings.HF_API_TOKEN
)

