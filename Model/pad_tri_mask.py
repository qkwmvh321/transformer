import torch
import numpy as np

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
    tri_tensor = torch.triu(torch.ones(seq_len, seq_len)).cuda()
    return tri_tensor