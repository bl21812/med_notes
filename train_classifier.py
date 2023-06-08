from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from accelerate import infer_auto_device_map, init_empty_weights

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")

print(tokenizer)
print(model)

test_input = ['The patient has bronchitis']

embed = model.model.embed_tokens

print(embed)

tokens = tokenizer(test_input)
print(tokens)

input_ids = torch.tensor(tokens['input_ids'])
# attention_mask = torch.tensor(tokens['attention_mask'])

# Project tokens to latent space
embeddings = embed(input_ids)
print(embeddings)

# decoder (attention) - take 10 of 40 units
num_units = 10
attention_units = model.model.layers
latents = attention_units.__getitem__(slice(num_units))
print(latents)

# i think i need some attention to get more info than just these lookup embeddings