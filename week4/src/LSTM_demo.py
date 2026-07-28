import torch

# 1. 准备数据
batch_size = 8
max_len = 20
vocab_size = 100
input_ids = torch.randint(low=1,high=vocab_size,
                          size=(batch_size,max_len),
                          dtype=torch.long)
length = torch.tensor([10,11,12,13,14,15,16,17])

import torch.nn as nn
embedding_dim = 10
padding_idx = 0
embedding = nn.Embedding(vocab_size,embedding_dim,
                         padding_idx)

embedded = embedding(input_ids)

# 还是告诉LSTM不需要计算padding，防止计算浪费以及它影响最终隐藏状态
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
pack_padded_data = pack_padded_sequence(input=embedded,lengths=length,
                                        batch_first=True,enforce_sorted=False)


# 2.定义模型
hidden_size = 8
lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,
               num_layers=3,batch_first=True)

'''返回值output, (hidden, cell),还需要解码output'''
pack_padded_output, (hidden, cell) = lstm(pack_padded_data)

# 要把长度参差不齐的隐藏状态持平到max_len，不够的补0
last_layer_hidden,length = pad_packed_sequence(pack_padded_output,batch_first=True,
                                        padding_value=padding_idx,
                                        total_length=input_ids.size(1))

print("output:", last_layer_hidden.shape)  # [8, 20, 8] batch_size,max_len,hidden_size最后一层 LSTM 在每个时间步产生的隐藏状态
print("hidden:", hidden.shape)  # [3, 8, 8] num_layer,batch_size,hidden_size每一层每句话最终的隐藏状态
print("cell:", cell.shape)      # [3, 8, 8] 每一层每句话的最终长期记忆ct