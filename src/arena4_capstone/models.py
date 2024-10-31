from arena4_capstone.util import device, settings

import torch as t
from nnsight import LanguageModel
from transformers import AutoTokenizer

import huggingface_hub as hf

hf.login(token=settings.HF_API_TOKEN)


if device.type == "cuda":
    t.cuda.empty_cache()

# It was necessary at some point to initialize the tokenizer manually, 
# but now it doesn't seem to be necessary.

# gemma_tokenizer = AutoTokenizer.from_pretrained(
#     "google/gemma-2-9b-it", token=settings.HF_API_TOKEN
# )

gemma = LanguageModel(
    "google/gemma-2-2b-it", device_map=device, token=settings.HF_API_TOKEN
)
