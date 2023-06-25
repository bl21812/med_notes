import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QA_Head(torch.nn.Module):

    def __init__(self, fc_layers=1, input_dims=(5120 * 5), output_dims=32001, hidden_dims=1024):

        super().__init__()
        self.fc_layers = torch.nn.Sequential()
        
        # NOTE: fix dims if I'm gonna use this
        for i in range(fc_layers-1):
            in_features = hidden_dims
            if i == 0:
                in_features = 14 * 5120  # output dims of LLM attention units (flattened)
            self.fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=hidden_dims, device='cuda:1'))
            self.fc_layers.append(torch.nn.ReLU())  # relu

        # Output head
        # set bias to false to match llama mlp head
        self.fc_layers.append(torch.nn.Linear(in_features=input_dims, out_features=output_dims, device='auto', bias=False))

    
    def forward(self, x):
        '''
        :param x: Flat latent-space embedding
        :return: Flat token tensor - predicted output sequence
        '''
        return self.fc_layers(x)


# ADD SPEAKER IDENTIFIER ?
# if it doesn't work too well - that could simplify the problem
class QA_Model(torch.nn.Module):

    def __init__(self, llm_vers, num_units=20, fc_layers=1, hidden_dims=1024):

        super().__init__()

        # pre-trained attention units
        alpaca = AutoModelForCausalLM.from_pretrained(llm_vers, device_map="auto")

        self.embed = alpaca.model.embed_tokens
        self.attention_units = alpaca.model.layers.__getitem__(slice(num_units))

        self.norm = alpaca.model.norm

        # MLP 
        # NOTE: if i use this may have to change dims ?
        self.fc_layers = torch.nn.Sequential()
        for i in range(fc_layers-1):
            in_features = hidden_dims
            if i == 0:
                in_features = 14 * 5120  # output dims of LLM attention units (flattened)
            self.fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=hidden_dims, device='cuda:1'))
            self.fc_layers.append(torch.nn.ReLU())  # relu

        # Output head
        self.fc_layers.append(alpaca.lm_head)

    def forward(self, x):
        
        # no fine-tuning for LLM layers
        with torch.no_grad():

            embeddings = self.embed(x)

            latents = embeddings
            for unit in self.attention_units:
                latents = unit(latents)[0]  # because the output is a tuple

        latents = self.norm(latents)

        # latents = torch.flatten(latents, start_dim=1)  # keep batch dim
        output = self.fc_layers(latents)

        return output
