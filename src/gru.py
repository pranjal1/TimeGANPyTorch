import torch
import torch.nn as nn


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.fc.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, lengths=T, batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(X_packed)
        out, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len,
        )
        out = self.activation(self.fc(out))
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden
