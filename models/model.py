import torch
from torch import nn
from models.attention import Seq_Transformer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        hidden_channel = 128

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, hidden_channel, kernel_size=1,
                      stride=configs.stride, bias=False, padding=0),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=1, stride=2, padding=1),  # (84-2+2)/2+1=43
            # nn.Dropout(configs.dropout)
        )

        self.num_channels = configs.final_out_channels
        self.lsoftmax = nn.LogSoftmax(1)
        # self.device = device

        self.projection_head = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1, stride=1, bias=False, padding=0),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(400, 100),
            nn.ReLU(),

        )

        self.concat = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Dropout(configs.dropout)
        )

        self.seq_transformer = Seq_Transformer(patch_size=101, dim=400, depth=1,
                                               heads=8, mlp_dim=200)

    def forward(self, data):

        features = self.conv_block1(data)

        c_t = self.seq_transformer(features)

        z = self.projection_head(c_t).squeeze(dim=1)

        yt = self.concat(z).squeeze(dim=1)

        return self.lsoftmax(yt)
        # return  x
