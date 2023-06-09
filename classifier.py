import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Classifier(torch.nn.Module):

    def __init__(self, llm_vers, num_units=10, fc_layers=1, hidden_dims=1024, num_classes=5):

        super().__init__()

        # pre-trained LLM
        tokenizer = AutoTokenizer.from_pretrained(llm_vers, device_map="auto")
        alpaca = AutoModelForCausalLM.from_pretrained(llm_vers, device_map="auto")

        self.tokenizer = tokenizer
        self.embed = alpaca.model.embed_tokens
        self.attention_units = alpaca.model.layers.__getitem__(slice(num_units))

        # MLP
        self.fc_layers = torch.nn.Sequential()
        for i in range(fc_layers-1):
            in_features = hidden_dims
            if i == 0:
                in_features = 5120  # output dims of LLM attention units
            self.fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=hidden_dims, device='cuda:1'))
            self.fc_layers.append(torch.nn.ReLU())  # relu
        
        # output FC
        in_features = hidden_dims
        if not self.fc_layers:
            in_features = 5120
        self.fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=num_classes, device='cuda:1'))
        self.fc_layers.append(torch.nn.Softmax(dim=1))  # softmax

    def forward(self, x):
        
        # no fine-tuning for LLM layers
        with torch.no_grad():

            tokens = self.tokenizer(x)
            input_ids = torch.tensor(tokens['input_ids'])

            embeddings = self.embed(input_ids)

            latents = embeddings
            for unit in self.attention_units:
                print(latents)
                print(latents.size())
                latents = unit(latents)[0]  # because the output is a tuple

        output = self.fc_layers(latents)

        return output

# why are fc layers on different devices ?
