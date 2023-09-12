import torch

from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer_source = "distilbert-base-uncased"
feature_extractor_source = "distilbert-base-uncased"
num_hidden_layers = 3
hidden_dim = 256

data_source = ".csv"  # UPDATE WITH CLASS DATASET

input_key = 'note'
label_key = 'class'

seed = 0
val_prop = 0.1

save_path = 'soap_class/C0001'

# ----- MODEL LOADING -----
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_source)
feature_extractor = DistilBertModel.from_pretrained(feature_extractor_source)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = feature_extractor(**encoded_input)

print(encoded_input)
print('\n\n\n\n\n')
print(output)
