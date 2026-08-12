import torch
import torch.nn as nn
import math

'''1.Multi-head Attention'''
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads,head_dim):
        super().__init__()

        assert num_heads*head_dim == embedding_dim

        self.input_to_q = nn.Linear(embedding_dim,embedding_dim)
        self.input_to_k = nn.Linear(embedding_dim,embedding_dim)
        self.input_to_v = nn.Linear(embedding_dim,embedding_dim)
        self.output_to_o = nn.Linear(embedding_dim,embedding_dim)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim


    def split_heads(self,matrix):
        '''matrix:(batch_size,max_len,embedding_dim)'''
        batch_size = matrix.size(0)
        max_len = matrix.size(1)

        matrix = matrix.view(batch_size,max_len,self.num_heads,self.head_dim)
        matrix = matrix.transpose(1,2)
        return matrix

    
    '''同时为self-attention和cross-attention服务'''
    def forward(self,input_q,input_k,input_v,mask):
        batch_size = input_q.size(0)
        max_len = input_q.size(1)
        '''input_q/k/v:(batch_size,max_len,embedding_dim)'''

        q = self.input_to_q(input_q)
        k = self.input_to_k(input_k)
        v = self.input_to_v(input_v)
        '''q/k/v:(batch_size,max_len,embedding_dim)'''    

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        '''q/k/v:(batch_size,num_heads,max_len,head_dim)'''

        match_score = torch.matmul(q,k.transpose(-1,-2))
        '''q*k^T:(batch_size,num_heads,max_len,max_len)'''

        scaled_match_score = match_score / math.sqrt(self.head_dim)

        if mask is not None: 
            '''mask:(batch_size,1,1,max_len)'''
            scaled_match_score = scaled_match_score.masked_fill(mask==0,float("-inf"))

        attention_weight = torch.softmax(scaled_match_score,dim=-1)
        '''attention_weight:(batch_size,num_heads,max_len,max_len)'''

        output = torch.matmul(attention_weight,v)
        '''(batch_size,num_heads,max_len,head_dim)'''

        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size,max_len,self.embedding_dim)
        output = self.output_to_o(output)

        return attention_weight,output


'''Position Embedding'''
class PositionEncoding(nn.Module):

    def __init__(self, embedding_dim,max_len):
        super().__init__()

        assert max_len % 2 == 0 # 必须max_len是偶数，好分组 
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        '''position(max_len,1)
        [
            [1],
            [2],
            [3],
            [4]
        ]'''

        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32)* (-math.log(10000.0) / embedding_dim))
        '''每组的频率(1,组数)
        [0.1,1]'''

        '''position*div_term:(max_len,组数)各位置各组数的频率'''

        position_encoding = torch.zeros(max_len, embedding_dim)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        '''position_encoding[:, 0::2]:(max_len,组数)'''

        position_encoding = position_encoding.unsqueeze(0)
        '''position_encoding:(1,max_len,embedding_dim)'''

        self.register_buffer("position_encoding", position_encoding)
        '''
        通常 requires_grad = False
        ↓
        不参与训练
        ↓
        optimizer 不更新'''

    def forward(self, x):
        return x + self.position_encoding


'''FFN'''
class FeedForwardNetwork(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)


    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


'''输入：已Token/Position Embedding后的x
流程：Multi-head Attention -> LayerNorm(残差连接) -> FFN -> LayerNorm(残差连接)'''
class EncoderBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,head_dim,hidden_dim,dropout):
        super().__init__()
        self.self_multi_head_attention = MultiHeadAttention(embedding_dim,num_heads,head_dim)
        self.ffn = FeedForwardNetwork(embedding_dim,hidden_dim)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x,mask):
        attention_weight,attention_output = self.self_multi_head_attention(input_q=x,input_k=x,input_v=x,mask=mask)

        '''经典 Transformer：Post-Norm'''
        x = self.layernorm1(x + self.dropout(attention_output))

        ffn = self.ffn(x)

        '''Post-norm:x=LayerNorm(x+Attention(x))
        Pre-norm:x=x+LayerNorm(Attention(x))'''
        x = self.layernorm2(x + self.dropout(ffn))
        return x,attention_weight


'''输入：Decoder截止目前输入的embedding后的x,以及encoder_output
流程：Multi-head Attention融合自身 -> LayerNorm(残差连接)消化自身特征 -> Cross-attention -> LayerNorm(残差连接)
FFN -> LayerNorm(残差连接)多了cross-attention，为了融合decoder output'''
class DecoderBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,head_dim,drop_out,hidden_dim):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim,num_heads,head_dim)
        self.cross_attention = MultiHeadAttention(embedding_dim,num_heads,head_dim)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.layernorm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(drop_out)
        self.ffn = FeedForwardNetwork(embedding_dim,hidden_dim)


    def forward(self,x,self_attetion_mask,cross_attetion_mask,encoder_output):
        '''x是截止目前Decoder输入的内容，比如中文译文
        x:(batch_size,decoder_len,embedding_dim)'''

        '''1.先进行self-attention，先融合其他token的信息'''
        self_attention_weight,self_attention_output = self.self_attention(
            input_q=x,input_k=x,input_v=x,mask=self_attetion_mask
        )
        '''batch_size,decoder_len,embedding'''

        '''2.再进行post-norm，加工融合后的信息'''
        x = self.layernorm1(x + self.dropout(self_attention_output))

        '''截止目前：我自己已经生成了什么，
        因为这里的self-attention是为了融合已经生成的token，所以mask是一个下三角矩阵
        实际意义就是看不到自身以后得token'''

        cross_attention_weight,cross_attention_output = self.cross_attention(
                    input_q=x,input_k=encoder_output,input_v=encoder_output,
                    mask=cross_attetion_mask
                )
        '''cross_attention_output:(batch_size,decoder_len,max_len)'''
        '''根据我已经生成到这里，原文中我现在应该关注什么'''
        
        x = self.layernorm2(x + self.dropout(cross_attention_output))

        ffn = self.ffn(x)

        x = self.layernorm3(x + self.dropout(ffn))

        return x,self_attention_weight,cross_attention_weight


class Encoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,max_len,dropout,num_heads,head_dim,hidden_dim,num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
        self.position_embedding = PositionEncoding(embedding_dim,max_len)
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dim,num_heads,head_dim,hidden_dim,dropout)
            for i in range(num_layers)
        ])


    def forward(self,input_ids,mask):
        # Token Embedding
        x = self.token_embedding(input_ids)

        # 原始 Transformer 会乘 sqrt(d_model)
        x = x * math.sqrt(self.embedding_dim)

        # Position Embedding
        x = self.position_embedding(x)
        x = self.dropout(x)

        self_attention_weights_list = []
        for encoder_block in self.layers:
            x,self_attention_weight = encoder_block(x,mask)
            self_attention_weights_list.append(self_attention_weight)

        return x,self_attention_weights_list


class Decoder(nn.Module):
    def __init__(self,vocab_size,embedding_dim,max_len,dropout,num_heads,head_dim,hidden_dim,num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
        self.position_embedding = PositionEncoding(embedding_dim,max_len)
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(embedding_dim,num_heads,head_dim,dropout,hidden_dim)
            for i in range(num_layers)
        ])


    def forward(self,input_ids,self_attetion_mask,cross_attetion_mask,encoder_output):
        x = self.token_embedding(input_ids)

        # 原始 Transformer 会乘 sqrt(d_model)
        x = x * math.sqrt(self.embedding_dim)

        # Position Embedding
        x = self.position_embedding(x)
        x = self.dropout(x)

        self_attention_weights_list = []
        cross_attention_weights_list = []

        for decoder_block in self.layers:
            x,self_attention_weight,cross_attention_weight = decoder_block(
                x,self_attetion_mask,cross_attetion_mask,encoder_output)
        
            self_attention_weights_list.append(self_attention_weight)
            cross_attention_weights_list.append(cross_attention_weight)

        return x,self_attention_weights_list,cross_attention_weights_list


class ClassicTransformer(nn.Module):
    def __init__(self,encoder_vocab_size,decoder_vocab_size,embedding_dim,
                 max_len,dropout,num_heads,hidden_dim,num_layers,padding_id):
        '''encoder_vocab_size理解成中文词典大小 decoder_vocab_size是英文词典大小
        max_len是中文的句子长度 max_len是英文句子长度'''

        assert embedding_dim % num_heads == 0
        head_dim = embedding_dim // num_heads #返回整数

        super().__init__()
        self.padding_id = padding_id
        self.encoder = Encoder(encoder_vocab_size,embedding_dim,
                               max_len,dropout,num_heads,
                               head_dim,hidden_dim,num_layers)
        self.decoder = Decoder(decoder_vocab_size,embedding_dim,
                               max_len,dropout,num_heads,
                               head_dim,hidden_dim,num_layers)
        self.decoder_output_to_output = nn.Linear(embedding_dim,
                                                  decoder_vocab_size)


    def padding_mask(self,input_ids):
        '''input_ids:(batch_size,max_len)'''

        judge_padding = input_ids != self.padding_id
        '''judge_padding:(batch_size,max_len)，元素是0/1'''

        judge_padding = judge_padding.unsqueeze(1).unsqueeze(2)
        '''judge_padding:(batch_size,1,1,max_len)
        因为encoder中的Q*K^T是(batch_size,num_heads,max_len,max_len)
        所以可以广播judge_padding来屏蔽padding'''

        return judge_padding


    def padding_causal_mask(self,input_ids):
        max_len = input_ids.size(1)

        judge_padding = input_ids != self.padding_id
        judge_padding = judge_padding.unsqueeze(1).unsqueeze(2)

        '''取一个下三角矩阵，屏蔽后文'''
        causal_mask = torch.tril(
            torch.ones(max_len,max_len,dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        '''causal_mask(1,1,max_len,max_len)
        因为decoder中self-attention的Q*K^T是
        (batch_size,num_heads,max_len,max_len)
        所以可以通过广播causal_mask来屏蔽后文'''

        return judge_padding & causal_mask


    def forward(self,encoder_input_ids,decoder_input_ids):
        encoder_self_attetion_mask = self.padding_mask(encoder_input_ids)
        decoder_self_attetion_mask = self.padding_causal_mask(decoder_input_ids)
        decoder_cross_attention_mask = self.padding_mask(encoder_input_ids)
        '''因为decoder中cross-attention的Q*K^T是
        (batch_size,num_heads,max_len,max_len)
        而且采用padding mask所以可以使用
        和encoder中相同的padding mask'''

        encoder_output,encoder_self_attention_weights_list = self.encoder(
            encoder_input_ids,encoder_self_attetion_mask)
        '''encoder_output:[batch, max_len, embedding_dim]'''

        decoder_output,decoder_self_attention_weights_list,decoder_cross_attention_weights_list = self.decoder(
            decoder_input_ids,decoder_self_attetion_mask,
                     decoder_cross_attention_mask,encoder_output)

        output = self.decoder_output_to_output(decoder_output)
        '''output:[batch, max_len, decoder_vocab_size]'''

        return {
            "logits": output,
            "encoder_output": encoder_output,
            "decoder_output": decoder_output,
            "encoder_self_attention_weights_list": encoder_self_attention_weights_list,
            "decoder_self_attention_weights_list": decoder_self_attention_weights_list,
            "decoder_cross_attention_weights_list": decoder_cross_attention_weights_list,
        }


if __name__ == "__main__":

    torch.manual_seed(15)

    embedding_dim = 12
    num_heads = 3
    hidden_dim = 48

    encoder_vocab_size = 100
    decoder_vocab_size = 120

    # ========================================================
    # Encoder 输入
    #
    # 第一条：
    # I love basketball PAD PAD
    #
    # 第二条：
    # You like playing basketball PAD
    # ========================================================
    encoder_input_ids = torch.tensor([
        [11, 25, 36, 0, 0, 0],
        [14, 27, 31, 36, 0, 0],
    ])

    # ========================================================
    # Decoder 输入
    #
    # 例如：
    #
    # <BOS> 我 喜欢 篮球
    # ========================================================
    decoder_input_ids = torch.tensor([
        [1, 41, 52, 63, 0, 0],
        [1, 42, 53, 63, 0, 0],
    ])

    max_len = 6
    dropout=0.0
    num_layers=2
    padding_id = 0
    model = ClassicTransformer(encoder_vocab_size,decoder_vocab_size,
                               embedding_dim,max_len,dropout,num_heads,
                               hidden_dim,num_layers,padding_id)

    result = model(encoder_input_ids, decoder_input_ids)

    print("encoder_input_ids:", encoder_input_ids.shape)
    print("decoder_input_ids:", decoder_input_ids.shape)

    print("encoder_output:", result["encoder_output"].shape)
    print("decoder_output:", result["decoder_output"].shape)
    print("logits:", result["logits"].shape)

    print(
        "encoder_self_attention_weights_list:",
        result["encoder_self_attention_weights_list"][0].shape,
    )

    print(
        "decoder_self_attention_weights_list:",
        result["decoder_self_attention_weights_list"][0].shape,
    )

    print(
        "decoder_cross_attention_weights_list:",
        result["decoder_cross_attention_weights_list"][0].shape,
    )