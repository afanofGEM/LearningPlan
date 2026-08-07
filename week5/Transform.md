1. 为什么需要首先对input_ids:(batch_size,max_len)进行
    TokenEmbedding + PositionEmbedding:
    
    没有位置编码时，如果句子只是对相同token进行了重新排序，那么某个token**对其他特定token的注意力权重不会因为顺序改变而改变**；attention matrix只是随输入顺序发生相同的置换。

2. Transformer前半段流程已经可以连起来了：

    Inputs IDs→ Token Embedding + Position Embedding
    → X → Q,K,V → MultiHeadAttention(attention_weight,output)