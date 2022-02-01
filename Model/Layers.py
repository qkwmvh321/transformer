import torch.nn as nn
import torch
from SubLayers import Multihead_Attention, Position_Wise_Feed_Forward_Layer, ResNet_Block


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
        
        print('outu1')
        muli_out = self.multi_attention_model(Q=input_data, K=input_data, V=input_data, en_mask=mask,de_mask=None)
        
        output1 = self.res_block(input_data,muli_out)
        print('out2')
        position_out = self.position_feedforward(output1)
        output2 = self.res_block(output1, position_out)
        
        return output2

    
    
#need to modify    
class Decoder_Layer(nn.Module):
    
    def __init__(self, batch_size, embedding_size, heads):
        super(Decoder_Layer, self).__init__()
        
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.heads = heads
        
        #trimask가 여기서 들어가야하나?
        self.masked_multi_attention = Multihead_Attention(self.batch_size, self.heads, self.embedding_size)
        self.multi_head_attention = Multihead_Attention(self.batch_size, self.heads, self.embedding_size)
        
        self.ffnn = Position_Wise_Feed_Forward_Layer(self.embedding_size)
        self.res_block = ResNet_Block(self.embedding_size)
        
    def forward(self, input_data, encoder_out, mask, src_mask):
        
        
        masked_muti_atten_out = self.masked_multi_attention(Q=input_data, K=input_data, 
                                                            V=input_data, mask=mask)
        
        output1 = self.res_block(input_data,masked_muti_atten_out)
        
        multi_head_atten_out = self.multi_head_attention(Q=masked_muti_atten_out, K=encoder_out,
                                                         V=encoder_out, mask=src_mask)
        
        output2 = self.res_block(output1, multi_head_atten_out)
        
        ffnn_out = self.ffnn(output2)
        
        output3 = self.res_block(output2,ffnn_out)
        
        return output3