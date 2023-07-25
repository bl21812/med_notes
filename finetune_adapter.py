from transformers.adapters import AdapterConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_source = "knkarthick/meeting-summary-samsum"
base_model_source = "knkarthick/meeting-summary-samsum"

model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_source, 
    device_map='auto'
)

print(model)

input()
