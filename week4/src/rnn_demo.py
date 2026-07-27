import torch

'''用MLP时常见做法就是先把序列维度聚合掉,比如平均/最大池化,
   得到batch_size,embed_dim再接全连接;
   而RNN会直接吃batch_size,max_len,embed_dim,自己沿着时间维度维护隐藏状态,不需要你先把序列合成一个向量'''

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


torch.manual_seed(42)

# =========================================================
# 1. 准备一批文本的 token ID
# =========================================================

PAD_ID = 0
vocab_size = 20
embedding_dim = 8
hidden_size = 6
num_classes = 5

# 3 条文本，统一补齐到 max_length=5
input_ids = torch.tensor(
    [
        [2, 5, 7, 9, 3],   # 实际长度 5
        [4, 6, 8, 1, 0],   # 实际长度 4
        [10, 11, 12, 0, 0] # 实际长度 3
    ],
    dtype=torch.long,
)

lengths = torch.tensor([5, 4, 3])

print("1. DataLoader 输出的 token ID")
print("input_ids.shape:", input_ids.shape)
print(input_ids)

# input_ids.shape:
# [batch_size, max_length]
# [3, 5]


# =========================================================
# 2. Embedding：token ID 转为词向量
# =========================================================

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=PAD_ID,
)

embedded = embedding(input_ids)

print("\n2. Embedding 后")
print("embedded.shape:", embedded.shape)

# embedded.shape:
# [batch_size, max_length, embedding_dim]
# [3, 5, 8]

'''input_size规定rnn一次处理一个Token中的embedding_dim个元素
   比如“我爱RNN”中的“我”
   hidden_size规定rnn将维度从embedding_dim变为hidden_size
   num_layers表示rnn网络循环几次：
    第一层的rnn接受的输入是(batch_size,max_len,embedding_dim)，
    从第二层开始rnn接受的输入是(batch_size,max_len,hidden_size)
   batch_first表示始终把batch_size作为第一维度 '''
rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,
             num_layers=2,batch_first=True,
             nonlinearity="tanh") # 规定输入数据的第一位是batch_size

'''告诉rnn,padding代表的embedding'''
packed_input = pack_padded_sequence(embedded,lengths,batch_first=True,
                                    enforce_sorted=False)


'''初始隐藏状态h0如果没有手动提供，PyTorch 会自动使用全零张量'''
'''packed_output保存每一条文本序列所有时间步的输出，比如
   序列1：h0,h1,h2,h3,h4,h5
   序列2: h0,h1,h2,h3,h4
   序列3：h0,h1,h2,h3'''
'''hidden保存的是每一条文本每一层的最终隐藏状态
   hidden(num_layers,batch_size,hidden_size)
   保存的是每一句话在每一层的最后一个隐藏状态，同样也不会算padding'''
packed_output, hidden = rnn(packed_input)

'''total_length表示max_len'''
rnn_output, output_length = pad_packed_sequence(packed_output,batch_first=True,
                                                padding_value=0,
                                                total_length=embedded.size(1))
'''rnn_output:每条文本在每个时间步的状态
   （batch_size,max_len,hidden_size)很巧的用max_len表示时间步的上限'''

rnn_classifier = nn.Linear(in_features=hidden_size,out_features=num_classes)
rnn_logits = rnn_classifier(hidden[-1]) # 最后一层中所有文本序列的最后隐藏状态
'''hidden[-1] (batch_size,hidden_size)'''
