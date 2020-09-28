# Import Modules
import math
import random
import numpy as np
from sru import SRU

# Import PyTorch
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

class model(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, n_layers, dropout):
        super(model, self).__init__()
        self.src_input_linear = nn.Linear(2, d_model)
        self.trg_input_linear = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(dropout)
        # Transformer
        # self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        # self.encoders = nn.ModuleList([
        #     TransformerEncoderLayer(d_model, self_attn, dim_feedforward,
        #         activation='gelu', dropout=dropout) for i in range(n_layers)])
        # encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, 
        #                                             dropout=dropout, activation='gelu')
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.transformer_ = nn.Transformer(d_model=d_model, nhead=n_head,
                                           num_encoder_layers=n_layers,
                                           num_decoder_layers=n_layers,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, activation='gelu',)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, sequence, target):
        sequence = sequence.transpose(0, 1)
        target = target.transpose(0, 1).unsqueeze(2)
        output = self.src_input_linear(sequence)
        target = self.trg_input_linear(target)
        # for i in range(len(self.encoders)):
        #     output = self.encoders[i](output)
        # output = self.transformer_encoder(output, src_key_padding_mask=None)
        output = self.transformer_(output, target)
        output = self.output_linear(self.dropout(F.gelu(output)))
        return output.transpose(0, 1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src