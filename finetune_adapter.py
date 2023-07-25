from transformers.adapters import ParallelConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_source = "knkarthick/meeting-summary-samsum"
base_model_source = "knkarthick/meeting-summary-samsum"

model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_source, 
    device_map='auto'
)

config = ParallelConfig(
    mh_adapter=True,
    output_adapter=True,  # can keep both of these in for now (unsure if needed)
    reduction_factor=16,  # important param !! (not sure what val)
    non_linearity="relu"
)
model.add_adapter("bottleneck_adapter", config=config)

print(model.config.adapters)

# model.train_adapter()
# print trainable params

input()
