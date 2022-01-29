import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer

from Embedding_Layer import Embedding_ready as embedding_layer


def get_mask(x,seq_len,heads):
        
    mask = x==12975
    remask = torch.repeat_interleave(mask,seq_len, dim=0).view(-1,seq_len,seq_len).view(-1,seq_len,seq_len)

    for idx in remask:
        for i,iidx in enumerate(idx[0]):
            if iidx ==True:
                break
        idx[i:]=True

    remask2=torch.repeat_interleave(remask.unsqueeze(1), heads,dim=1)
    return remask2


class Encoder(nn.Module):

    def __init__(self, seq_len,batch_size, embedding_size, heads, n_layer):
        super(Encoder,self).__init__()
        
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.heads = heads
        self.batch_size = batch_size
        self.n_layer = n_layer 
        
        self.src_embedding_encoder = Embedding_ready(30000,self.embedding_size, self.seq_len, self.heads)
        self.dropout = nn.Dropout(0.1)
        
        self.layer_stack = nn.ModuleList([Encoder_Layer(self.batch_size, self.embedding_size, self.heads) for i in range(self.n_layer)])
        
        self.layer_norm = nn.LayerNorm(self.embedding_size, eps=1e-6)
                
    def forward(self, input_data, mask):

        enc_slf_attn_list = []

        embedding_data, mask = self.src_embedding_encoder(input_data)

        enc_output = self.dropout(embedding_data)
        enc_output = self.layer_norm(enc_output)

        for encoder_layer in self.layer_stack:
            
            embedding_data = encoder_layer(embedding_data, mask)

        return embedding_data