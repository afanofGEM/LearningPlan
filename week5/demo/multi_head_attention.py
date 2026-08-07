import torch 
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads,head_dim):
        super().__init__()
        self.input2q = nn.Linear(in_features=embedding_dim,out_features=embedding_dim)
        self.input2k = nn.Linear(in_features=embedding_dim,out_features=embedding_dim)
        self.input2v = nn.Linear(in_features=embedding_dim,out_features=embedding_dim)
        self.output2o = nn.Linear(in_features=embedding_dim,out_features=embedding_dim)
        self.num_heads = num_heads
        self.head_dim = head_dim

        
    def split_heads(self,matrix):
        batch_size = matrix.size(0)
        max_len = matrix.size(1)
        matrix = matrix.view(batch_size,max_len,self.num_heads,self.head_dim)

        # 要把heads提前，到时候按照head划分
        matrix = matrix.transpose(1,2)
        return matrix 
    '''(batch_size,num_heads,max_len,head_dim)
    num_heads * head_dim = embedding_dim'''


    def rope(self,matrix):
        '''matrix(q,k):(batch_size,num_heads,max_len,head_dim)'''
        batch_size = matrix.size(0)
        num_heads = matrix.size(1)
        max_len = matrix.size(2)
        head_dim = matrix.size(3)
        assert head_dim % 2 ==0

        position = torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
        '''position:(max_len,1)
        [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5]
        ]'''

        '''为不同位置的不同二维组(h0,h1),(h2,h3)...(..,h_head_dim-1)设置不同的旋转角度
        先划分频率
        (h0,h1) → 都是快钟
        (h2,h3) → 都是慢钟
        再异化位置
        所以需要position*frequency来确定每个位置的不同维度组的旋转角度
        比如位置1的(h0,h1)和位置2的(h0,h1)旋转的角度不一样'''
        frequency = torch.exp(torch.arange(0,head_dim,2,dtype=torch.float32)* 
                             (-math.log(10000.0)/ head_dim))
        '''[0.1,1]第一组快钟转频为0.1，第二组慢钟转频为1'''

        angle = position * frequency
        '''(max_position,1) * (1,组数)：(max_position,组数)
        不同位置各组的旋转角度'''

        cos_angle = torch.cos(angle).unsqueeze(0).unsqueeze(0)
        sin_angle = torch.sin(angle).unsqueeze(0).unsqueeze(0)
        '''cos/sin_angle:(1,1,max_position,组数)，奇偶数维都这么旋转'''
        '''会把Q K的维度表示拆成奇数维度和偶数维度
        每一个拆分矩阵(batch_size,num_heads,max_position,组数)
        但是会有2个这样的矩阵
        Q1：(h1,h3) Q2:(h2,h4)
        可以直接和angle运算，因为刚好一个奇偶数维组的旋转角度相同'''

        even_matrix = matrix[...,0::2]
        odd_matrix = matrix[...,1::2]
        rotated_even = (even_matrix*cos_angle-odd_matrix*sin_angle)
        rotated_odd = (even_matrix*sin_angle+odd_matrix*cos_angle)
        '''rotated_odd/even:(batch_size,num_heads,max_position,组数)'''

        rotated = torch.stack((rotated_even,rotated_odd,),dim=-1)
        rotated = rotated.flatten(start_dim=-2)
        '''(batch_size,num_heads,max_position,head_dim)完成合并'''
        return rotated
    

    def forward(self,input_ids,judge_padding):

        batch_size = input_ids.size(0)
        max_len = input_ids.size(1)
        embedding_dim = input_ids.size(2)
        '''input_ids:(batch_size,max_len,embedding_dim)'''

        q = self.input2q(input_ids)
        k = self.input2k(input_ids)
        v = self.input2v(input_ids)
        '''q,k,v:(batch_size,max_len,embedding_dim)'''

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        '''q,k,v:(batch_size,num_heads,max_len,head_dim)'''
        q = self.rope(q)
        k = self.rope(k)

        matched_score = torch.matmul(q,k.transpose(-2,-1))
        '''(batch_size,num_heads,max_len,head_dim)与
        (batch_size,num_heads,head_dim,max_len)
        结果(batch_size,num_heads,max_len,max_len)
        会有num_heads个匹配分数的矩阵'''

        scaled_matched_score = matched_score / math.sqrt(self.head_dim)
        '''(batch_size,num_heads,max_len,max_len)'''

        judge_padding = judge_padding.unsqueeze(1).unsqueeze(2)
        '''(batch_size,max_len)到
        (batch_size,1,1,max_len)在(1,1)上都进行广播'''

        scaled_matched_score = scaled_matched_score.masked_fill(
            judge_padding == 0, float('-inf')
        )

        attention_weight = torch.softmax(scaled_matched_score,dim=-1)
        '''(batch_size,num_heads,max_len,max_len)对每行的各列进行softmax,
        使得每行每个Token对所有Token的注意力为1'''

        output = torch.matmul(attention_weight,v)
        '''(batch_size,num_heads,max_len,head_dim)'''

        # 进行多头合并
        output = output.transpose(1,2)
        '''(batch_size,max_len,num_heads,head_dim)'''

        output = output.contiguous().view(batch_size,max_len,embedding_dim)
        '''(batch_size,max_len,embedding_dim)'''

        output = self.output2o(output)
        return attention_weight,output


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 6
    embedding_dim = 12
    num_heads = 3
    head_dim = int(embedding_dim / num_heads)

    x = torch.randn(batch_size,seq_len,embedding_dim)

    padding_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    attention = MultiHeadSelfAttention(embedding_dim=embedding_dim,num_heads=num_heads,
                                       head_dim=head_dim)

    attention_weights,output  = attention(x,padding_mask)