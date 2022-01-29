import torch
import torch.nn as nn

__author__ = 'sanguk Han'

class PositionalEncoding(nn.Module): #잘 되는거

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        self.dropout = nn.Dropout(0.1)
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        y= x + self.pos_table[:, :x.size(1)]
        y2 = self.dropout(y)
        
        return y2.clone().detach()

class Embedding(nn.Module):
    
    def __init__(self,max_vocab_len, embedding_size):
        super(Embedding, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=max_vocab_len,
                                            embedding_dim=embedding_size)
        
        self.max_vocab_len = max_vocab_len
        self.embedding_size = embedding_size
        
    def forward(self,x):
        
        y = self.embedding_layer(x)
        return y
    
    
class Embedding_ready(nn.Module):
    
    def __init__(self,max_vocab_len, embeding_size, seq_len, heads):
        super(Embedding_ready,self).__init__()
        
        self.embedding_layer = Embedding(max_vocab_len,embeding_size)
        self.position_encoder = PositionalEncoding(embeding_size)
        self.seq_len = seq_len
        self.heads = heads
        
    def get_mask(self,x):
        
        mask = x==12975
        remask = torch.repeat_interleave(mask,self.seq_len, dim=0).view(-1,self.seq_len,self.seq_len).view(-1,self.seq_len,self.seq_len)
        
        for idx in remask:
            for i,iidx in enumerate(idx[0]):
                if iidx ==True:
                    break
            idx[i:]=True
            
        remask2=torch.repeat_interleave(remask.unsqueeze(1),self.heads,dim=1)
        return remask2
        
    def forward(self,x):
        print(x.shape)
        mask =self.get_mask(x)
        x_embedding  = self.embedding_layer(x)
        x_pos_embedding = self.position_encoder(x_embedding)
        
        return x_pos_embedding,mask