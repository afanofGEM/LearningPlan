import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim,max_len):
        super().__init__()
        self.tokenEmbedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=embedding_dim)
        self.positionEmbedding = nn.Embedding(num_embeddings=max_len,
                                              embedding_dim=embedding_dim)

    def forward(self,input_ids):
        '''input_ids:(batch_size,max_len)'''
        batch_size = input_ids.size(0)
        max_len = input_ids.size(1)

        position_ids = torch.arange(max_len)
        '''position_ids:(max_len)'''
        position_ids = position_ids.unsqueeze(0)
        '''position_ids:(1,max_len)''' 
        position_embedded = self.positionEmbedding(position_ids)
        '''position_embedded:(1,max_len,embedding_dim)'''  

        token_embedded = self.tokenEmbedding(input_ids)
        '''token_embedded:(batch_size,max_len,embedding_dim)'''

        transformer_inputs = position_embedded + token_embedded
        '''transformer_inputs:(batch_size,max_len,embedding_dim)
        就是输入的x'''

        return transformer_inputs


torch.manual_seed(42)

batch_size = 2
seq_len = 6
d_model = 12
vocab_size = 100

token_ids = torch.randint(low=0,high=vocab_size,size=(batch_size,seq_len,))

embedding_layer = TransformerEmbedding(vocab_size=vocab_size,embedding_dim=d_model,
                                       max_len=seq_len)

transformer_input = embedding_layer(token_ids)

print("token_ids shape:",token_ids.shape,)
print("transformer_input shape:",transformer_input.shape,)
