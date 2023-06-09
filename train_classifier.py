from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .classifier import Classifier

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")
alpaca = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")

classifier = Classifier(tokenizer, alpaca)

test_input = 'hi guys i am a doctor and this is a renal issue'

output = classifier(test_input)
print(output)

# Load data 

# Train loop