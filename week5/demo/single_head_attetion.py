import torch
import math

def scaled_dot_attention(Q,K,V,judge_padding):
    '''Q: (batch_size, max_len, embedding_dim)
       K: (batch_size, max_len, embedding_dim) 他俩必须一样，要相乘
       V: (batch_size, max_len, embedding_dim)
        dk dv并不一定等于embedding_dim，因为多头注意力就不相等
       judge_padding(batch_size,max_len),判断哪些位置是padding，
       设置它为负无穷，这样softmax后就是0
    '''

    '''matmul矩阵乘法，transpose T
        得到匹配分数(Token之间的注意力)'''
    match_scores = torch.matmul(Q,K.transpose(-2,-1))  #K:(batch_size,max_len,embedding_dim) -> (batch_Size,embedding_dim,max_len)

    '''缩放，防止softmax分布不均，梯度消失'''
    scaled_match_scores = match_scores / math.sqrt(Q.size(-1)) # Q.size(-1):embedding_dim
    '''scaled_match_scores(batch_size,max_len,max_len) 一个batch_size就是一句话
     
    scaled_match_scores:
                        被关注的 Key
                我的  手机  突然  没有  信号  了
    Query 我的    a    b    c    d    e    f    这里的每一行都是V中各行的一套组合方式
    Query 手机    g    ·    ·    ·    ·    ·    每一行是某个Token对所有Token的关注度
    Query 突然    ·    ·    ·    ·    ·    ·
    Query 没有    ·    ·    ·    ·    ·    ·
    Query 信号    ·    ·    ·    ·    ·    ·
    Query 了      ·    ·    ·    ·    ·    · 
    
    V
            embedding_dim1 embedding_dim2 embedding_dim3 embedding_dim4
    我的     ·               ·               ·              ·    row1
    手机     ·               ·               ·              ·    row2
    突然     ·               ·               ·              ·    row3
    没有     ·               ·               ·              ·    row4
    信号     ·               ·               ·              ·    row5
    了       ·               ·               ·              ·    row6
    
    SV:
        row1 = a*row1 + b*row2 + c*row3 + d*row4 + e*row5 + f*row6
        row2 = g*row1 + ....
    
                    embedding_dim1 embedding_dim2 embedding_dim3 embedding_dim4
    我的     ·               ·               ·              ·    row1
    手机     ·               ·               ·              ·    row2
    突然     ·               ·               ·              ·    row3
    没有     ·               ·               ·              ·    row4
    信号     ·               ·               ·              ·    row5
    了       ·               ·               ·              ·    row6
    '''

    judge_padding = judge_padding.unsqueeze(1) 
    '''(batch_size,1,max_len)需要按行对padding归零
    [
        [[1,1,1,1,1,1,0]],
        [[1,0,1,1,0,1,0]]
    ] 为什么可以直接通过0来置换为-inf，使得softmax的值为0，
      因为match_scores的每一行也是按照文本顺序排列的'''

    '''tensor.masked_fill(condition,value)
    含义是：对 condition 为 True 的位置，用 value 替换原来的数值
    judge_padding在运算时会按行广播至(max_len,max_len)'''
    scaled_match_scores = scaled_match_scores.masked_fill(judge_padding==0,
                                                          float('-inf'))

    # 对最后一维，也就是每列的数据softmax,使得每行的数据和为1(某个Token对本句话所有Token关注为1)
    attention_weight = torch.softmax(scaled_match_scores,dim=-1)

    output = torch.matmul(attention_weight,V)
    '''output:(batch_size, max_len, embedding_dim)'''

    return attention_weight,output


def main():
    torch.manual_seed(15)

    batch_size = 3
    max_len = 10
    embedding_dim = 8

    x =  torch.randn(batch_size,max_len,embedding_dim)
    Wq = torch.randn(embedding_dim,embedding_dim)
    Wk = torch.randn(embedding_dim,embedding_dim)
    Wv = torch.randn(embedding_dim,embedding_dim)
    q = torch.matmul(x,Wq)
    k = torch.matmul(x,Wk)
    v = torch.matmul(x,Wv)

    judge_padding = torch.tensor(
    [
        [1,1,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,0,0,0]
    ])

    attention_weight,output = scaled_dot_attention(q,k,v,judge_padding)

    print("X 形状：",x.shape)
    print("Q形状：",q.shape)
    print("K 形状：",k.shape)
    print("V 形状：",v.shape)
    print("注意力权重(匹配分数)形状：",attention_weight.shape)
    print("输出形状：",output.shape,)
    print("\n第一条工单中，" "第5个token的注意力权重(关注权重)：")
    print(attention_weight[0,4,:])
    print("\n这组权重之和：",attention_weight[0,4,:].sum())


if __name__ == "__main__":
    main()
