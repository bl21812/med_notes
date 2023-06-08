from transformers import AutoTokenizer, AutoModelForCausalLM
# from accelerate import infer_auto_device_map, init_empty_weights

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")

print(tokenizer)
print(model)

test_sent = 'The patient has bronchitis'

embed = model.model.embed_tokens

print(embed)

tokens = tokenizer(test_sent)
print(tokens)

output = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
print(output)

latents = embed(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
print(latents)