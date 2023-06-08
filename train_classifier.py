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
attention_mask = torch.tensor(tokens['attention_mask'])

output = model(input_ids=input_ids, attention_mask=attention_mask)
print(output)

latents = embed(input_ids=input_ids, attention_mask=attention_mask)
print(latents)