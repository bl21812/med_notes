from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")

print(tokenizer)
print(model)

test_sent = 'This is a sentence with cardiovascular issues'

embed = model.model.embed_tokens

print(embed)

