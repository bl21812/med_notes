import torch


class Classifier(torch.nn.Module):

    def __init__(self, tokenizer, alpaca, num_units=10, fc_layers=1, hidden_dims=1024, num_classes=5):

        # pre-trained LLM
        self.tokenizer = tokenizer
        self.embed = alpaca.model.embed_tokens
        self.attention_units = alpaca.model.layers.__getitem__(slice(num_units))

        # MLP
        self.fc_layers = torch.nn.Sequential()
        for i in range(fc_layers-1):
            in_features = hidden_dims
            if i == 0:
                in_features = 5120  # output dims of LLM attention units
            self.fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=hidden_dims))
            self.fc_layers.append(torch.nn.ReLU())  # relu
        
        # output FC
        in_features = hidden_dims
        if not self.fc_layers:
            in_features = 5120
        self.fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=num_classes))
        self.fc_layers.append(torch.nn.Softmax(dim=1))  # softmax

        # don't fine-tune LLM layers
        for param in self.tokenizer.params:
            param.requires_grad = False
        for param in self.embed.params:
            param.requires_grad = False
        for unit in self.attention_units:
            for param in unit.params:
                param.requires_grad = False

    def forward(self, x):
        
        tokens = self.tokenizer(x)
        input_ids = torch.tensor(tokens['input_ids'])

        embeddings = self.embed(input_ids)

        latents = embeddings
        for unit in self.attention_units:
            latents = unit(latents)

        output = self.fc_layers(latents)

        return output
