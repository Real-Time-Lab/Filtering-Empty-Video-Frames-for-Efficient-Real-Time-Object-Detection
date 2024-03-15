import torch
import torch.nn as nn
from torchsummary import summary


class LSTM_model(nn.Module):  ## lr=0.001, epoch=20
    def __init__(self):
        super(LSTM_model, self).__init__()
        self.lstm2 = nn.LSTM(2, 64, 1, batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(64, 16, 1, batch_first=True, dropout=0.2)

        self.out = nn.Linear(16, 1)

    def forward(self, x):
        '''
            h0(num_layers * num_directions, batch, hidden_size), e.g. (1,10,128)
            c0(num_layers * num_directions, batch, hidden_size)
        '''
        r_out, (hn2, cn2) = self.lstm2(x)
        r_out, _ = self.lstm3(r_out)
        r_out = r_out[:, -1, :]  ##(batch size, seq, hidden size)

        out = nn.functional.relu(r_out)
        out = self.out(out)
        return out


model = LSTM_model()
model
