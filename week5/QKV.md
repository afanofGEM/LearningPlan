1. 符号含义：
    1.1. 每个 token 都同时有 Q、K、V

    句子：我的 手机 突然 没有 信号 了
    每个 token 先有一个输入向量：**尺寸均为(1,embedding_dim)**
    x_我的
    x_手机
    x_突然
    x_没有
    x_信号
    x_了

    然后每个 token 都会经过三个不同的线性变换：
    qi=xi * WQ  **均不改变尺寸(1,embedding_dim)**
    ki=xi * WK
    vi=xi * WV

    1.2. Q和K：
    qi：表示此Token需要向外找些什么，比如 **“信号”** token，单纯的理解token本身是mlp的水平，
        它还需要在Token向量中融入上下文的信息，比如 **信号有没有，哪个设备的信号，没了多长时间了，在哪里没有信号**
        qi = [0.2,0,-0.3,0.5] 
        是否寻找否定信息
        是否寻找设备信息
        是否寻找时间状态
        是否寻找地区状态

    ki: 表示此Token的自身特征，比如它自身表示的是 **信号/设备特征/时间特征/地域特征**
        ki = [0,0.9,-0.8,0.3],比如这样可能就表示设备特征

    1.3. Q与K：
    现在每个Token的qi会与每句文本的所有Token的ki做点积：
        q信号 * k我的
        q信号 * k手机
        q信号 * k突然
        q信号 * k没有
        q信号 * k信号
        q信号 * k了     **结果是一个数**
    点积**越大**，说明“信号”想找的特征与ki所具有的特征**越匹配**
                    匹配分数
        我的          0.2
        手机          1.8
        突然          0.7
        没有          2.4
        信号          1.5
        了            0.1
    更新“信号”时，应该重点看“没有”和“手机”。

    1.4. V：
    vi目前还不知道什么含义，可能是用于运算产生最新表示的token
    原先的QK结果：
                    匹配分数
        我的          0.2
        手机          1.8
        突然          0.7
        没有          2.4
        信号          1.5
        了            0.1
    先经过softmax:
                匹配分数
        我的          0.03
        手机          0.25
        突然          0.07
        没有          0.40
        信号          0.20
        了            0.05

    最终“信号”的新表示为：
            o信号 = 0.03v我的 + 0.25v手机 + 0.07v突然 + 0.40v没有 + 0.20v信号 + 0.05v了 **(1,embedding_dim)**


2. 计算过程：
    1. 计算匹配程度 **Q * K^T**
        Q                       [2, 6, 8]
        K.transpose(-1, -2)     [2, 8, 6]
        ---------------------------------
        attention_scores        [2, 6, 6]
        
        所以会得到一张 6 × 6 的匹配分数表。
                        被关注的 Key
                    我的  手机  突然  没有  信号  了

        Query 我的    ·    ·    ·    ·    ·    ·
        Query 手机    ·    ·    ·    ·    ·    ·
        Query 突然    ·    ·    ·    ·    ·    ·
        Query 没有    ·    ·    ·    ·    ·    ·
        Query 信号    ·    ·    ·    ·    ·    ·
        Query 了      ·    ·    ·    ·    ·    ·  
        **例如第 5 行：attention_scores[0, 4, :]表示第一批文本的第一条工单中，“信号”这个Query对6个token的匹配分数**
    
    2. 除以sqrt(dk)
        假设ki,qi中的每一个元素服从N(0,1),那么每一个点积sigma(qi*ki)的均值为0，方差为dk
        **所以除以sqrt(dk)是为了让每个元素服从N(0,1),即除以embedding_dim**
        **如果不服从N(0,1)，softmax后的值会及其离谱**
        
        对每一行而言：softmax([1, 2, 3])大约得到：[0.09, 0.24, 0.67] 还算平滑。
        但：softmax([10, 20, 30])大约得到：[0.000000002, 0.000045, 0.999955] 几乎变成只关注最后一个位置。

        这会导致两个问题：
            注意力分布过于极端
            Softmax附近的梯度容易变小，训练不稳定 **其导数为softmax * (1-softmax)**


    3. ![alt text](image.png)


3.第七部分：Attention的尺寸推导总结

需要能不看答案自己写出下面这张表：

张量	            尺寸	                                   含义
X	        [batch_size, max_len, embedding_dim]	    输入 token 表示
W_Q	        [embedding_dim, embedding_dim]              Query 投影矩阵
W_K	        [embedding_dim, embedding_dim]	            Key 投影矩阵
W_V	        [embedding_dim, embedding_dim]	            Value 投影矩阵
Q	        [batch_size, max_len, embedding_dim]	每个token对各特征(上下文)的搜索意愿
K	        [batch_size, max_len, embedding_dim]	   每个token的特征属性
V	        [batch_size, max_len, embedding_dim]	                   
Kᵀ	        [batch_size, embedding_dim, max_len]	                   
QKᵀ	        [batch_size, max_len, max_len]	         token两两之间的匹配分数(注意力)
weights	    [batch_size, max_len, max_len]	           除以d_k,Softmax后的注意力权重
output	    [batch_size, max_len, embedding_dim]	   融合上下文后的 token 表示
weights * output