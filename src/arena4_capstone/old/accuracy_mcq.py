#%%
import gc
import json
from nnsight import LanguageModel
from sympy import pretty_print
import torch as t
from tqdm import tqdm
from transformers import AutoTokenizer

from arena.plotly_utils import imshow
from dataset_mcq import MCQDataset, project_root, mcq_questions

#%%


API_TOKEN = open(project_root / "token.txt").read()

!export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 
t.cuda.empty_cache()

device = t.device("cuda" if t.cuda.is_available() else "cpu")

honest_dataset = MCQDataset(mcq_questions[:-10], mode="prepend_honest")
lying_dataset = MCQDataset(mcq_questions[:-10], mode="prepend_lying")
# test_dataset = MCQDataset(mcq_questions[-10:], mode="none")
with open(project_root / "datasets" / "mcq.json") as f:
    mcq_questions = json.load(f)["data"]


# %%
test_dataset = MCQDataset(mcq_questions, mode="none")


#%%

# It's necessary to load the tokenizer explicitly, for some reason
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it", use_auth_token=API_TOKEN
)

gemma = LanguageModel("google/gemma-2-9b-it", device_map=device, token=API_TOKEN)

#%%
def completions_accuracy(corrects: list[str], predictions: list[str]) -> float:
    assert len(corrects) == len(predictions)
    total_correct = 0
    total_answers = 0
    for correct, predicted in zip(corrects, predictions):
        predicted = predicted.strip().upper()
        if predicted in ["A", "B"]: 
            total_answers += 1
            if correct.strip().upper() == predicted:
                total_correct += 1

    return total_correct / total_answers


def continue_text(model, prompt):
    with model.generate(max_new_tokens=50) as generator:
        with generator.invoke(prompt):
            for n in range(50):
                model.next()
            all_tokens = model.generator.output.save()

    complete_string = model.tokenizer.batch_decode(all_tokens.value)[0]

    return complete_string

@t.inference_mode()
def accuracy_on_dataset(dataset: MCQDataset, model):
    full_completions = []
    full_completions_decoded = []

    for batch in range(0, dataset.size, 10):
        prompt_token_sequences = model.tokenizer(dataset.prompts[batch : batch + 10])[
            "input_ids"
        ]
        with model.trace(prompt_token_sequences):
            all_tokens = model.lm_head.output[..., -1, :].argmax(axis=-1).save()

        completion_tokens = all_tokens.value
        completions_decoded = model.tokenizer.batch_decode(
            completion_tokens[:, None], skip_special_tokens=False
        )

        full_completions_decoded.extend(completions_decoded)
        full_completions.extend(dataset.completions[batch : batch + 10])


        for i, (prompt, completion) in enumerate(zip(dataset.prompts, completions_decoded)):
            print(f"Prompt: {prompt}")
            print(f"Predicted next token: {completion}")
            print("-" * 50)

    return completions_accuracy(full_completions, full_completions_decoded)


@t.inference_mode()
def last_token_batch_mean(model, dataset: MCQDataset):
    saves = {} # batch, layer -> tensor of dimension (batch_size, d_model)
    for batch in range(0, dataset.size, 10):
        with model.trace(dataset.prompts[batch : batch + 10]):
            for i, layer in enumerate(gemma.model.layers):
                saves[batch//10, i] = layer.output[0][:, -1, :].save()

    n_batches = len(dataset.prompts) // 10

    out = t.stack([
        t.concatenate([saves[batch, layer].value for batch in range(n_batches)], dim=0)
        for layer in range(model.config.num_hidden_layers)
    ], dim=0)

    assert out.shape == (model.config.num_hidden_layers, n_batches, model.config.hidden_size)

    return out.mean(dim=1)


@t.inference_mode()
def last_token_mean(model, dataset: MCQDataset):
    saves = [] # list of tensors, each of dimension (num_hidden_layers, d_model)
    for prompt in tqdm(dataset.prompts):
        with model.trace([prompt]):
            layer_outputs = []
            for layer in model.model.layers:
                layer_outputs.append(layer.output[0][0, -1, :].save())
        saves.append(t.stack([output.value for output in layer_outputs]))

    out = t.stack(saves)

    assert out.shape == (len(dataset.prompts), model.config.num_hidden_layers, model.config.hidden_size)

    return out.mean(dim=0)

#%%

lying_accuracy = (
    accuracy_on_dataset(
        model=gemma, dataset=lying_dataset
    )
)

honest_accuracy = (
    accuracy_on_dataset(
        model=gemma, dataset=honest_dataset
    )
)

lying_accuracy, honest_accuracy

# %%

# python garbage collecector clear all stuff in gpu
# gc.collect()
# # clear cuda memory
# t.cuda.empty_cache()

lying_vectors = last_token_mean(gemma,lying_dataset)
# %%

honest_vectors = last_token_mean(gemma, honest_dataset)

# %%

intervention_vector = lying_vectors - honest_vectors


@t.inference_mode()
def intervene(model, dataset: MCQDataset, vectors: t.Tensor):
    """
    Intervene on the model by adding the vector to the last layer of the model.

    vectors should be of shape (num_hidden_layers, d_model)
    """

    prompts = dataset.prompts
    # first run the model without intervention
    

    # now for each layer, run the model once with the intervention vector
    

    honest_ids = t.tensor(model.tokenizer(dataset.completions, add_special_tokens=False)["input_ids"], device=device).squeeze()
    print(f"honest_ids.shape={honest_ids.shape}")

    lying_completions = [{"A": "B", "B": "A"}[completion] for completion in dataset.completions]
    lying_ids = t.tensor(model.tokenizer(lying_completions, add_special_tokens=False)["input_ids"], device=device).squeeze()

    print(honest_ids.shape)
    print(lying_ids.shape)

    print(prompts)

    prompt_token_sequences = t.tensor(model.tokenizer(prompts, padding=True)["input_ids"], device=device)
    print(f'{prompt_token_sequences.shape=}')
    with t.inference_mode():
        with model.trace() as tracer:
            with tracer.invoke(prompt_token_sequences):
                shape = model.lm_head.output.shape.save()
                # something = model.lm_head.output[0][0,0,0].save()
                honest_logits_without_intervention = model.lm_head.output.log_softmax(dim=-1)[t.arange(len(honest_ids)), -1,  honest_ids].save()
                lying_logits_without_intervention = model.lm_head.output.log_softmax(dim=-1)[t.arange(len(lying_ids)), -1, lying_ids].save()
                prediction_without_intervention = model.lm_head.output.log_softmax(dim=-1)[:, -1,  :].argmax(axis=-1).save()

    print("output shape:", shape.value)
    honest_logits_without_intervention = honest_logits_without_intervention.value
    lying_logits_without_intervention = lying_logits_without_intervention.value
    prediction_without_intervention = prediction_without_intervention.value

    
    honest_logits_with_intervention = t.zeros(model.config.num_hidden_layers, len(honest_ids), device=device)
    lying_logits_with_intervention = t.zeros(model.config.num_hidden_layers, len(lying_ids), device=device)
    prediction_with_intervention = t.zeros(model.config.num_hidden_layers, len(honest_ids), device=device)

    for i, layer in enumerate(model.model.layers):
        with model.trace(prompts):
            layer.output[0][:, -1, :] += vectors[i]

            # Index into the logits at the honest_ids
            honest_logits = model.lm_head.output.log_softmax(dim=-1)[t.arange(len(honest_ids)), -1, honest_ids].save()
            lying_logits = model.lm_head.output.log_softmax(dim=-1)[t.arange(len(lying_ids)), -1, lying_ids].save()
            prediction = model.lm_head.output.log_softmax(dim=-1)[:, -1,  :].argmax(axis=-1).save()

        honest_logits_with_intervention[i] = honest_logits.value
        lying_logits_with_intervention[i] = lying_logits.value
        # print(f"prediction_with_intervention.shape={prediction_with_intervention.shape}")
        # print(f"lying_logits_with_intervention.shape={lying_logits_with_intervention.shape}")
        # print(f"prediction.shape={prediction.value.shape}")
        prediction_with_intervention[i] = prediction.value

    return honest_logits_with_intervention, lying_logits_with_intervention, prediction_with_intervention, honest_logits_without_intervention, lying_logits_without_intervention, prediction_without_intervention

results = intervene(gemma, test_dataset, intervention_vector)


h_l_w_i, l_l_w_i, p_w_i, h_l_wo_i, l_l_wo_i, p_wo_i = results
# %%
# imshow(
#     (results[1] - results[4]).mean(dim=-1).unsqueeze(-1),
#     title = "Average indirect effect of function-vector intervention on antonym task",
#     width = 1000,
#     height = 600,
#     labels = {"x": "batch", "y": "layer"},
#     aspect = "equal",
# )
# %%
print(test_dataset.completions)
# %%
imshow(
   t.log(((l_l_w_i - h_l_w_i) / (l_l_wo_i - h_l_wo_i))),#.mean(dim=-1).unsqueeze(-1),
    title = "Average indirect effect of function-vector intervention on lying task",
    width = 1000,
    height = 600,
    labels = {"x": "batch", "y": "layer"},
    aspect = "equal",
)
# %%
imshow(
   ((l_l_w_i - h_l_w_i) - (l_l_wo_i - h_l_wo_i)),#.mean(dim=-1).unsqueeze(-1),
    title = "Average indirect effect of function-vector intervention on lying task",
    width = 1000,
    height = 600,
    labels = {"x": "batch", "y": "layer"},
    aspect = "equal",
)
#%%
num = (l_l_w_i - h_l_w_i).mean(dim=-1).unsqueeze(-1)
denom = (l_l_wo_i - h_l_wo_i).mean(dim=-1).unsqueeze(-1)
imshow(
   t.log(num/denom),#.mean(dim=-1).unsqueeze(-1),
    title = "Average indirect effect of function-vector intervention on lying task",
    width = 1000,
    height = 600,
    labels = {"x": "batch", "y": "layer"},
    aspect = "equal",
)

# %%
@t.inference_mode()
def intervene_with_h(model, dataset: MCQDataset, h: t.Tensor, layer: int, coefficient: float):
    """
    Intervene on the model by adding the vector to the last layer of the model.

    vectors should be of shape (num_hidden_layers, d_model)
    """

    with model.trace(dataset.prompts):
        model.model.layers[layer].output[0][:, -1, :] += coefficient * h
        log_probs = model.lm_head.output.log_softmax(dim=-1)[:, -1, :]
        prediction = log_probs.argmax(dim=-1).save()
    # Print each prompt and its prediction
    for prompt, pred in zip(dataset.prompts, prediction.value):
        predicted_token = model.tokenizer.decode(pred.item())
        print(f"Prompt: {prompt}")
        print(f"Predicted next token: {predicted_token}")
        print("-" * 50)

    predictions = model.tokenizer.batch_decode(prediction.value[:, None], skip_special_tokens=False)
    return predictions

# %%
layer= 24
accuracy = []
for i in range(-6, 6):
    predictions = intervene_with_h(gemma, test_dataset, intervention_vector[layer], layer, float(i))
    accuracy.append(completions_accuracy(test_dataset.completions, predictions))

print(accuracy)
# %%
predictions = intervene_with_h(gemma, test_dataset, intervention_vector[layer], layer, 6.0)
# %%
def continue_text(model, prompt, h, coefficient):
    with model.generate(max_new_tokens=50) as generator:
        with generator.invoke(prompt):
            model.model.layers[layer].output[0][:, -1, :] += coefficient * h
            for n in range(50):
                model.next()
            all_tokens = model.generator.output.save()

    complete_string = model.tokenizer.batch_decode(all_tokens.value, skip_special_tokens=False)

    return complete_string
#%%
layer = 24
full_completions = (continue_text(gemma, test_dataset.prompts, intervention_vector[layer], 9.0))

for completion in full_completions:
    print(completion)
# %%
