import torch

# 1.准备数据
input_ids = torch.tensor(
    [
        [2, 5, 7, 9, 3],   # 实际长度 5
        [4, 6, 8, 1, 0],   # 实际长度 4
        [10, 11, 12, 0, 0] # 实际长度 3
    ],
    dtype=torch.long,
)
length = torch.tensor([5,4,3])

# 2.对数据embedding,不管是mlp还是rnn都需要
import torch.nn as nn
vocab_size = 32
embedding_dim = 10
padding_idx = 0
embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx)
embedded = embedding(input_ids)
'''embedded(batch_size,max_len,embedding_dim)'''

# 3.告诉rnn哪些是有效数据，它就不用处理padding了
from torch.nn.utils.rnn import (
   pack_padded_sequence,
   pad_packed_sequence,
)
pack_padded_data = pack_padded_sequence(input=embedded,lengths=length,batch_first=True,
                                      enforce_sorted=False)

# 4.构造rnn,构造cnn是通过构造不同的核完成的，但是rnn一气呵成
hidden_size = 6
rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,
             num_layers=3,batch_first=True,
             nonlinearity='tanh')

# 5.进入模型,隐藏状态与输出
all_hidden,last_hidden = rnn(pack_padded_data) # 得到个时间步的隐藏状态和最终的隐藏状态
'''all_hidden:最后一层的隐藏状态记录
   序列1：h0,h1,h2,h3,h4,h5
   序列2: h0,h1,h2,h3,h4
   序列3：h0,h1,h2,h3'''

'''last_hidden:每个文本序列每层最终的隐藏状态
   (num_layers,batch_size,hidden_size)'''

# 6.将all_hidden转化成tensor
rnn_hidden,output_length = pad_packed_sequence(all_hidden,batch_first=True,padding_value=padding_idx,
                    total_length=input_ids.size(1))
'''total_length就是要把每个序列的隐藏状态记录补0至max_len'''
'''rnn_hidden:(batch_size,max_len,hidden_size)
   output_length:每句文本的长度[5,4,3]'''

# 7.last_hidden经过分类器得到输出结果
num_classes = 5
rnn_classifier = nn.Linear(in_features=hidden_size,out_features=num_classes)
rnn_logits = rnn_classifier(last_hidden[-1]) # 取最后一层每句话的最终隐藏状态

# 做一些记录
print(f'原始数据尺寸：{input_ids.shape}')
print(f'embedding后数据尺寸：{embedded.shape}')
print(f'经过RNN后数据尺寸：{rnn_hidden.shape}')
print(f'最后一层最终隐藏状态尺寸：{last_hidden[-1].shape}')
print(f'输出数据尺寸：{rnn_logits.shape}')

# 单一的文本序列过程
print('单一的文本序列过程')
single_text = embedded[0:1] # 表示取第一句话(1,max_len,embedding_dim)

single_last_hidden = None
for i in range(single_text.size(1)):
   single_token = single_text[:,i:i+1,:] #(1,1,embedding_dim)
   all_hidden,single_last_hidden = rnn(single_token,single_last_hidden)
   '''all_hidden:序列1：h0,h1,h2....
      last_hidden:(num_layer,batch_size,hidden_size)'''

   print(f'last_hidden.shape:{single_last_hidden.shape}')


'''RNN
[3, 5, 8]
     ↓ 按时间步依次处理
[3, 5, 6]   output

最终 hidden：
[1, 3, 6]

取最后一层：
[3, 6]

     ↓ Linear
[3, 5类]'''

'''MLP:
[3, 5, 8]
     ↓ 平均池化，压缩 max_length
[3, 8]
     ↓ Linear
[3, 5类]'''