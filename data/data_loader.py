from torch.utils.data import Dataset
import torch.utils.data as data
from tokenizers import SentencePieceBPETokenizer
import json
import numpy as np
import math
import torch

#<pad>가 있는 부분에 mask를 씌운다.
def get_pad_mask(x,seq_len,heads):
        
    mask = x==4648
    
    remask = torch.repeat_interleave(mask,seq_len, dim=0).view(-1,seq_len,seq_len).view(-1,seq_len,seq_len)

    for idx in remask:
        for i,iidx in enumerate(idx[0]):
            if iidx ==True:
                break
        idx[i:]=True

    remask2=torch.repeat_interleave(remask.unsqueeze(1), heads,dim=1)
    return remask2

#tri 
def get_tri_mask(seq_len):
    tri_tensor = torch.triu(torch.ones(seq_len, seq_len))
    return tri_tensor


def load_json(data_name):
    data = json.load(open(data_name))
    return data

class data_loader(Dataset):
  
    def __init__(self,batch_size,seq_len,heads):
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.heads = heads
        #, preproc=None
        self.data_path = './' #data_path
        self.data_dict = load_json(self.data_path)['data']
        
        self.tokenizer = SentencePieceBPETokenizer(vocab_file='./tokenizer/vocab.json',
                          merges_file='./tokenizer/merges.txt',unk_token='<unk>')
        
        user_defined_symbols = ['[BOS]','[EOS]','[UNK0]','[UNK1]','[UNK2]','[PAD]',
                        '[UNK3]','[UNK4]','[UNK5]','[UNK6]','[UNK7]','[UNK8]','[UNK9]']
        self.tokenizer.add_special_tokens(user_defined_symbols)
        
                    
    def __len__(self):
        
        return len(self.data_dict)
    
    def get_token(self,data):
        
        sen_token = self.tokenizer.encode(data['sentence']).ids
        pa_token = self.tokenizer.encode(data['paraphrase']).ids

        if len(sen_token) != self.seq_len:
            sentence_token = [sen_token[i] if i<len(sen_token) else self.tokenizer.encode('[PAD]').ids[0] for i in range(self.seq_len) ]
        else:
            sentence_token = sen_token
        if len(pa_token) != self.seq_len:
            paraphrase_token = [pa_token[i] if i<len (pa_token) else self.tokenizer.encode('[PAD]').ids[0] for i in range(self.seq_len) ]
        else:
            paraphrase_token = pa_token
        
        return sentence_token, paraphrase_token
    
    
    #get image
    def __getitem__(self,idx):
        src_tensor = np.zeros([self.batch_size, self.seq_len])
        trg_tensor = np.zeros([self.batch_size, self.seq_len])
        
        for i,idx in  enumerate(self.data_dict[idx*self.batch_size : self.batch_size*(idx+1)]):
            src_tensor[i,:], trg_tensor[i,:] = self.get_token(idx)
        src_tensor = torch.tensor(src_tensor)
        trg_tensor = torch.tensor(trg_tensor)
        src_mask =get_pad_mask(src_tensor,self.seq_len,self.heads)
        
        trg_mask =get_pad_mask(trg_tensor,self.seq_len,self.heads)
        tri_mask = get_tri_mask(self.seq_len)
        tri_trg_mask = trg_mask.masked_fill(tri_mask==1.0,-1e9)
        
        
        return src_tensor, src_mask, trg_tensor, tri_trg_mask