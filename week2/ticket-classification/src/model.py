import torch
import torch.nn as nn

class TicketClassifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_classes,padding_idx=0):

        '''Embedding 把字符编号变成字符向量，
        mean pooling 把多个字符向量合成文本向量，
        Linear 再把文本向量变成每个类别的分数'''
        
        super().__init__()
        self.padding_idx = padding_idx  # padding_idx不参与mask mean pooling
        
        # 创建embedding参数矩阵 (vocab_size,embedding_dim)
        self.embedding = nn.Embedding(vocab_size,embedding_dim,self.padding_idx)

        # 创建线性层
        self.linear = nn.Linear(embedding_dim,num_classes)
    
    
    def forward(self,text_ids):

        '''text_ids=
        [
            [1,2,3,4,5],
            [0,7,8,9,10]
        ]'''

        # 正常的embedding编码，但是缺陷是把padding_id也编了
        # 这逼其实应该是0
        embedding_ids = self.embedding(text_ids)
        '''embedding_ids:(batch_size,max_len,embedding_dim)
        [
            [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]],
            [[0.1,0.2,0.3],[7,7,7],[8,8,8],[9,9,9],[10,10,10]]
        ]'''

        # ne = not equal
        judge_ids = text_ids.ne(self.padding_idx).unsqueeze(-1) # 最后加一个维度，一维
        '''judge_ids:(batch_size,max_len,1)
        [
            [[True],[True],[True],[True],[True]], true运算时会广播的true,true,true
            [[False],[True],[True],[True],[True]]
        ]'''

        # 这下真编完了，padding也变成0了
        embedding_ids = embedding_ids * judge_ids
        '''embedding_ids:(batch_size,max_len,embedding_dim)
        [
            [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]],
            [[0,0,0],[7,7,7],[8,8,8],[9,9,9],[10,10,10]]
        ]'''

        # 第二维度消失了，每一条文本的维度都是embedding_dim，所以只需要每个embedding求和
        sum_embedding_ids = embedding_ids.sum(dim=1) # 是的下标为1的维度消失
        '''[1,1,1]+[2,2,2]+[3,3,3]+[4,4,4]+[5,5,5]=[15,15,15]
        sum_embedding_ids:(batch_size,embedding_dim)
        [
            [15,15,15],
            [34,34,34]
        ]接下来需要除以每一条文本的有效Token字数，完成平均池化'''

        # dim=1就是求每行的所有列之和，dim=0就是求每列的所有行之和
        num_token = judge_ids.sum(dim=1).clamp(min=1) # 如果一条文本全是pad，记为1，防止除数为0
        '''num_token: (batch_size,1)
        [
            [5],
            [4]
        ]'''

        pooled = sum_embedding_ids / num_token
        '''pooled:(batch_size,embedding_dim)
        [
            [3,3,3],
            [8.5,8.5,8.5]
        ]'''

        output = self.linear(pooled)
        '''out_put:(batch_size,num_classes)'''
        return output


if __name__ == "__main__":
    model = TicketClassifier(vocab_size=200,embedding_dim=64,num_classes=5,padding_idx=0)

    fake_input_ids = torch.randint(low=0,high=200,size=(8, 32))

    logits = model(fake_input_ids)

    print("input_ids shape:", fake_input_ids.shape)
    print("logits shape:", logits.shape)

    