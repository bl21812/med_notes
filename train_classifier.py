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

# decoder (attention)
layers = model.model.layers
for layer in layers._modules:
    print(layer)

# i think i need some attention to get more info than just these lookup embeddings