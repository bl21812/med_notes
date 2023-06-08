from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights
from torchsummary import summary

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")

print(model)

# dummy utterance?