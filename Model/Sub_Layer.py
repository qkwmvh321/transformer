import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class Multihead_Attention(nn.Module):
    
    def __init__(self,batch_size,head_size, embedding_size):
        super(Multihead_Attention,self).__init__()
        
        self.batch_size = batch_size
        self.k_size = self.q_size =self.v_size = self.embedding_size = embedding_size
        self.head = head_size
        
        self.q_layer = nn.Linear(self.q_size, embedding_size*self.head)
        self.k_layer = nn.Linear(self.k_size, embedding_size*self.head)
        self.v_layer = nn.Linear(self.v_size, embedding_size*self.head)
        
        self.final_layer = nn.Linear(embedding_size * self.head, self.v_size)
        
    def forward(self,x, mask=None):
        
        query = self.q_layer(x)
        key = self.k_layer(x)
        value  = self.v_layer(x)
        
        #print('query_size : ',query)
        print('key_size : ',key.shape)
        print('value_size : ',value.shape)
        
        #멀티 헤드를 다시 나눠준다. layer를 통과한값은 정확히는 head * embeddingsize기때문에 이것을 다시 head 만큼 나눠준다.
        # 1024*8 = 8184  layer를 통과한 값은 (batch_size * seq_len * 8184(head * embedding size) )
        #이것을 다시 (batch size * head * seq_len * embedding size) 로 변경 시켜준다.
        query_view = query.view(query.size(0),query.size(1),self.head, self.q_size)
        key_view = key.view(key.size(0), key.size(1), self.head, self.k_size)
        value_view = value.view(value.size(0),value.size(1),self.head,self.v_size)
        
        query = query_view.transpose(1,2)
        key = key_view.transpose(1,2)
        value = value_view.transpose(1,2)
        
        key_T = key.transpose(2,3)
        
        print('query : ',query.shape)
        print('key_T : ',key_T.shape )
        #query와 key를 matmul을 한다 이때 key는 key_T로 변경해줘야 한다. matmul한 값은 루트 embeddingsize로 나눠서 scaling한다.
        atten_score = torch.matmul(query,key_T)/ math.sqrt(self.embedding_size)
        #print('attne_socre',atten_score.shape)
        
        if mask is not None:
            #큰 -값을 주면 softmax에서는 0값으로 반영 되어진다. 
            masked_atten_score = atten_score.masked_fill(mask==True, -1e9)
            
        soft_atten = F.softmax(masked_atten_score,dim=-1)
        attention = torch.matmul(soft_atten, value)
        attention = attention.transpose(1,2)
        attention = attention.contiguous().view(self.batch_size, -1,self.head*self.embedding_size)
        final_embedding = self.final_layer(attention)
        
        return final_embedding

    
class Position_Wise_Feed_Forward_Layer(nn.Module):
    
    def __init__(self,embedding_size):
        super(Position_Wise_Feed_Forward_Layer,self).__init__()
        
        self.embedding_size = embedding_size
    
        self.first_layer = nn.Linear(self.embedding_size, self.embedding_size*2 )
        self.second_layer = nn.Linear(self.embedding_size*2, self.embedding_size)
        
        self.relu = nn.ReLU()
        self.dropout= nn.Dropout(0.2)
    def forward(self,input_data,maks=None):
        
        dff = self.first_layer(input_data)
        dff = self.relu(dff)
        dff = self.dropout(dff)
        dff2 = self.second_layer(dff)
        
        return dff2