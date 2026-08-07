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