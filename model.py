import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            dropout,
            norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "bn":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "ln":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "bn":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "ln":
                    self.norms.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h = self.dropout(h).relu()
                if self.norm_type != "none":
                    h = self.norms[l](h)
        return h

class SAMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layer=1,
                 dropout=.5, norm_type='none', use_predictor=False):
        super().__init__()
        self.mlpA = nn.Linear(num_nodes, hidden_channels)
        self.mlpX = nn.Linear(in_channels, hidden_channels)
        self.atten = nn.Linear(hidden_channels * 2, 1)
        self.classifierA = MLP(hidden_channels, hidden_channels, out_channels, num_layer, dropout, norm_type=norm_type)
        self.classifierX = MLP(hidden_channels, hidden_channels, out_channels, num_layer, dropout, norm_type=norm_type)
        self.dropout = nn.Dropout(dropout)
        self.use_predictor = use_predictor
        self.latent_predictor = MLP(in_channels, hidden_channels, hidden_channels, 2, dropout, norm_type=norm_type)

    def decouple_encoder(self, A, X):
        if self.use_predictor:
            HA = self.latent_predictor(X)
        else:
            HA = self.mlpA(A)
        HA = self.dropout(HA).relu()

        HX = self.mlpX(X)
        HX = self.dropout(HX).relu()

        H = torch.cat([HA, HX], dim=1)
        return H, HA, HX

    def attentive_decoder(self, H, HA, HX):
        yA = self.classifierA(HA)
        yX = self.classifierX(HX)

        alpha = self.atten(H).sigmoid()
        y = yA * alpha.view(-1, 1) + yX * (1 - alpha.view(-1, 1))
        return y

    def forward(self, A, X):
        H, HA, HX = self.decouple_encoder(A, X)
        y = self.attentive_decoder(H, HA, HX)
        return y