import torch.nn as nn
import torch
from SubLayers import Multihead_Attention, Position_Wise_Feed_Forward_Layer


__author__ = "sanguk Han"

class Encoder_Layer(nn.Module):
    
    def __init__(self,batch_size, embedding_size, heads):
        super(Encoder_Layer, self).__init__()
        
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.heads = heads
        
        #self.Embedding_encoder  = Embedding_ready(30000, self.embedding_size, self.batch_size, self.heads)
        self.multi_attention_model = Multihead_Attention(self.batch_size, self.heads, self.embedding_size)
        self.position_feedforward = Position_Wise_Feed_Forward_Layer(self.embedding_size)
        
        self.res_block = ResNet_Block(self.embedding_size)
        
    def forward(self,input_data, mask):
        
        output1 = self.res_block(input_data, self.multi_attention_model, mask)
        
        output2 = self.res_block(output1, self.position_feedforward)
        
        return output2

    
    
#need to modify    
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self,batch_size, embedding_size, heads):
        super(DecoderLayer, self).__init__()
        
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.heads = heads
        
        self.slf_attn = Multihead_Attention(self.batch_size, self.heads, self.embedding_size)
        self.enc_attn = Multihead_Attention(self.batch_size, self.heads, self.embedding_size)
        self.position_feedforward = Position_Wise_Feed_Forward_Layer(self.embedding_size)

    def forward(self, dec_input, enc_output,slf_attn_mask=None, dec_enc_attn_mask=None):
        
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        
        dec_output = self.position_feedforward(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn