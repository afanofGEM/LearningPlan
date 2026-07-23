import torch.nn as nn

class classifier_mlp(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,padding_id,hidden_dim,num_classes,dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_id)
        self.padding_id = padding_id
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim,hidden_dim), #(batch_size,hidden_dim)
            nn.ReLU(),
            nn.Dropout(dropout), # 把隐藏层的输出结果一定程度置为0，防止过拟合
            nn.Linear(hidden_dim,num_classes)
        )

    
    def forward(self,text_ids):
        # text_ids:(batch_size,max_len)
        '''text_ids = 
        [
            [1,2,3],
            [0,5,6]
        ]'''
        embedding_text = self.embedding(text_ids)

        # embedding_text:(batch_size,max_len,embedding_dim)
        '''但是此时padding_id还被编成了非0
        embedding_text = 
        [
            [[1,1,1],[2,2,2],[3,3,3]],
            [[0.1,0.1,0.1],[5,5,5],[6,6,6]]
        ]'''

        judge_ids = text_ids.ne(self.padding_id).unsqueeze(-1) # T变成[T]，方便运算
        # judge_ids:(batch_size,max_len,1)
        '''judge_ids = 
        [
            [[T],[T],[T]],
            [[F],[T],[T]]
        ]'''

        # 真正的embedding编码
        embedding_text = embedding_text * judge_ids
        '''embedding_text = 
        [
            [[1,1,1],[2,2,2],[3,3,3]],
            [[0,0,0],[5,5,5],[6,6,6]]
        ]'''

        # 获得真正可以表示每一条文本的编码，脱离max_len限制
        # dim=1，让第二维度消失，对每一个字的embedding_dim个数据求和即可
        embedding_text = embedding_text.sum(dim=1)
        '''embedding_text = 
        [
            [6,6,6],
            [11,11,11]
        ]'''

        # 现在在进行池化，求平均值，计算每一条文本有多少真实Token
        count_num = judge_ids.sum(dim=1).clamp(min=1) # 在第二维方向堆积，最小结果也是1
        '''count_num=
        [
            [3],
            [2]
        ]'''

        embedding_text /= count_num
        '''embedding_text = 
        [
            [2,2,2],
            [5.5,5.5,5.5]
        ]  (batch_size,embedding_dim)'''

        result = self.mlp(embedding_text)
        '''(batch_size,num_classes)'''
        return result
    

if __name__ == '__main__':
    vocab_size = 1000
    embedding_dim = 64
    padding_id = 0
    hidden_dim = 128
    num_classes = 5
    dropout = 0.3
    model = classifier_mlp(vocab_size=vocab_size,embedding_dim=embedding_dim,
                           padding_id=padding_id,hidden_dim=hidden_dim,
                           num_classes=num_classes,dropout=dropout)
    
    import torch
    text_ids = torch.randint(low=1,high=vocab_size,size=(8,32))
    text_ids[0, 10:] = padding_id
    text_ids[1, 15:] = padding_id
    text_ids[2, 20:] = padding_id
    text_ids[3, 8:] = padding_id
    result = model(text_ids)
    print("text_ids shape:", text_ids.shape)
    print("logits shape:", result.shape)