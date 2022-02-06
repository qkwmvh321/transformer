import torch
import torch.nn as nn
import numpy as np
import attention.Layers as layers
import Embedding_Layer as embedding_layer


class Encoder(nn.Module):

    
    def __init__(self, batch_size,seq_len, embedding_size, heads, n_layer):
        super(Encoder,self).__init__()
        
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.heads = heads
        self.batch_size = batch_size
        self.n_layer = n_layer 
        
        
        self.src_embedding_encoder = embedding_layer.Embedding_ready(30000,self.embedding_size, self.seq_len, self.heads)
        self.dropout = nn.Dropout(0.1)
        
        self.layer_stack = nn.ModuleList([layers.Encoder_Layer(self.batch_size, self.embedding_size, self.heads) for i in range(self.n_layer)])
        
        self.layer_norm = nn.LayerNorm(self.embedding_size, eps=1e-6)

    def forward(self, src_data,src_mask):


        embedding_data = self.src_embedding_encoder(src_data)

        enc_output = self.dropout(embedding_data)
        enc_output = self.layer_norm(enc_output)

        for encoder_layer in self.layer_stack:
            embedding_data = encoder_layer(embedding_data, src_mask)

        return embedding_data

    

class Decoder(nn.Module):

    def __init__(self, batch_size, seq_len, embedding_size, heads, n_layer):
        super(Decoder,self).__init__()
        
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.heads = heads
        self.batch_size = batch_size
        self.n_layer = n_layer 
        
        self.trg_embedding_encoder = embedding_layer.Embedding_ready(30000,self.embedding_size, self.seq_len, self.heads)
        self.dropout = nn.Dropout(0.1)
        
        self.layer_stack = nn.ModuleList([layers.Decoder_Layer(self.batch_size, self.embedding_size, self.heads) for i in range(self.n_layer)])
        
        self.layer_norm = nn.LayerNorm(self.embedding_size, eps=1e-6)
        self.final_layer = nn.Linear(self.embedding_size, 30000)
        
    def forward(self,trg_data, trg_tri_mask, encoder_out, src_mask):

        para_em = self.trg_embedding_encoder(trg_data)
        
        para_em = self.dropout(para_em)
        para_em = self.layer_norm(para_em)
        
        for decoder_layer in self.layer_stack:
            
            para_em = decoder_layer(para_em, trg_tri_mask, encoder_out, src_mask)
        
        out = self.final_layer(para_em)
        
        #output = F.softmax(out)
        
        return out


class Transformer(nn.Module):
    
    def __init__(self,batch_size, seq_len, embedding_size, heads, n_layer):
        super(Transformer,self).__init__()
        
        self.batch_size = batch_size
        self.seq_len  = seq_len
        self.embedding_size = embedding_size 
        self.heads = heads
        self.n_layer = n_layer
        
        self.encoder = Encoder(self.batch_size,self.seq_len,
                               self.embedding_size,self.heads,
                               self.n_layer)
        
        self.decoder = Decoder(self.batch_size,self.seq_len,
                               self.embedding_size,self.heads, 
                               self.n_layer)
        
            
    
    def forward(self, src_data, src_mask,
                      trg_data, trg_tri_mask):
        
        encoder_out = self.encoder(src_data, src_mask)
        
        decoder_out = self.decoder(trg_data, trg_tri_mask, encoder_out, src_mask)
        
        return decoder_out
        
        