from arena4_capstone.models import gemma_2_2b_it

eos = str(gemma_2_2b_it.tokenizer.eos_token)
bos = str(gemma_2_2b_it.tokenizer.bos_token)
sot = "<start_of_turn>"
eot = "<end_of_turn>"
