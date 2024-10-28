from arena4_capstone.util import device, settings, REMOTE_MODE

import torch as t
from nnsight import LanguageModel
from transformers import AutoTokenizer


if device.type == "cuda":
    t.cuda.empty_cache()

gemma_tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it", token=settings.HF_API_TOKEN
)

if REMOTE_MODE:
    gemma = LanguageModel("google/gemma-2-9b-it", token=settings.HF_API_TOKEN)
else:
    gemma = LanguageModel(
        "google/gemma-2-9b-it", device_map=device, token=settings.NNSIGHT_API_TOKEN
    )
