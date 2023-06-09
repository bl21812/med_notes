from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .classifier import Classifier

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")
alpaca = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-13b", device_map="auto")

classifier = Classifier(tokenizer, alpaca)

# LOAD DATA AND TRAIN LOOP