1. Transformer概述：
    Transformer属于Sequence to Sequence模型，所以不仅仅
    像传统分类任务一样需要encoder理解输入序列，
    而且需要decoder生成输出序列

1. 为什么需要首先对input_ids:(batch_size,max_len)进行
    TokenEmbedding + PositionEmbedding:
    
    没有位置编码时，如果句子只是对相同token进行了重新排序，那么某个token**对其他特定token的注意力权重不会因为顺序改变而改变**；attention matrix只是随输入顺序发生相同的置换。


2. Transformer的Inputs IDs->MultiAttention流程：

    Inputs IDs(x)→ Token Embedding + Position Embedding
    → X → Q,K,V → MultiHeadAttention(attention_weight,output)


3. Transform Encoder Block:

    I love basketball
            ↓
    Token Embedding + Position Embedding
            ↓
    ┌────────────────────────────┐
    │ Encoder Block              │
    │                            │
    │ Self-Attention (multi-head attention)
    │ ↓                          │
    │ Residual + LayerNorm       │
    │ ↓                          │
    │ FFN                        │
    │ ↓                          │
    │ Residual + LayerNorm       │
    └────────────────────────────┘
            ↓
    再堆若干 Encoder Block
            ↓
    Encoder Output

残差连接：x = x + attention_output 尺寸均为(batch_size,max_len,embedding_dim)
    原始手机特征+从“我的、没有、信号”等词得到的信息
    而不是：直接把原始手机表示扔掉只留下Attention输出

LayerNorm:分别处理每个token的Embedding_dim个特征
    1. **标准化，将元素转化为N(0,1)**
    2. 再对每个元素进行yi​=γi*​x^i​+βi​，这么做的原因：
    因为强制所有数据永远：均值=0 方差=1 可能太死板。
    因此神经网络可以自己学习：
        “归一化是挺好的，但这个第 37 维我还是想稍微放大一点。”

FFN：
    [2, 6, 128] embedding_dim

    ↓ Linear

    [2, 6, 512] hidden_dim

    ↓ GELU

    [2, 6, 512]

    ↓ Linear

    [2, 6, 128]

**流程**
x→ 让token互相交流(multi-head attention)
→保留旧信息(残差连接)→稳定数值(LayerNorm)
→每个 token 自己加工(FFN)
→再保留旧信息(残差连接)→稳定数值(LayerNorm)


4. 由于经典Transformer出于解决翻译问题，Encoder解决理解英文的问题，Decoder解决生成中文的问题。


5. 推理时 Decoder 怎么工作？

Encode 一次性处理完整原文：
    I love basketball
            ↓
    Encoder
            ↓
    encoder_output

然后Decoder开始：

第一步Decoder输入：<BOS>
根据：<BOS>+encoder_output
预测：我

第二步：把刚才生成的「我」放回 Decoder：
输入：<BOS> 我
预测：喜欢

第三步
输入：<BOS> 我 喜欢
预测：篮球

第四步
输入：<BOS> 我 喜欢 篮球
预测：<EOS>

所以真正推理时是：
    <BOS>
    ↓
    我
    ↓
    喜欢
    ↓
    篮球
    ↓
    <EOS>

**自回归生成** ：把之前生成的结果继续作为下一步输入。


6. Decoder第一步：Cross-Attention
假如现在Encoder：
I love basketball
所以：
source_len = 3
Encoder Output：[1, 3, 128]

Decoder进行到预测<SOS>,所以输入为:
<BOS> 我 喜欢 篮球
所以：target_len = 4
Q.shape=[1, 4, 128]
K.shape=V.shape=[1, 3, 128]  **Cross-Attention表示Q,K,V来自不同的x**

QKᵀ:4 个 Decoder token，每一个都对 3 个 Encoder token 算注意力   
                    Encoder

                I    love   basketball

Decoder BOS     •      •        •

        我      •      •        •

        喜欢    •      •        •

        篮球    •      •        •


7. Decoder的流程：
Decoder 输入：<BOS> 我 喜欢 篮球(x)
        ↓
Masked Multi-Head
Self-Attention： Q,K,V来自输入x
        ↓
Residual + LayerNorm
        ↓
Cross-Attention

Q ← Decoder
K,V ← Encoder
        ↓   
Residual + LayerNorm
        ↓
FFN
        ↓
Residual + LayerNorm
        ↓
Decoder Block 输出


8. 为什么 Decoder 要先 Self-Attention，再 Cross-Attention？

假设已经生成：我 喜欢
Decoder 首先要知道：我目前已经说了什么？
所以先：Masked Self-Attention
形成：对当前目标语言上下文的理解
然后它才拿着这个状态去问 Encoder：根据我目前翻译到这里，原文哪里最值得看？
于是再：Cross-Attention

所以逻辑是：

    先看看自己已经写到哪
            ↓
    再回头看看原文
            ↓
    决定接下来怎么处理