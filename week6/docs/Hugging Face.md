Hugging Face是服务BERT的工具库

1. AutoTokenizer(WordPiece Tokenizer，既不按字也不按词切，playing->play##ing)
"手机没有信号"
↓ 分词
[CLS] 手 机 没 有 信 号 [SEP]
↓ 查词表
[101, xxxx, xxxx, xxxx, xxxx, xxxx, xxxx, 102]
↓ 转 Tensor
input_ids
attention_mask
token_type_ids

你之前是：
自己建 vocab
↓
自己 text → id
↓
nn.Embedding

现在 BERT 是：
官方 tokenizer
↓
input_ids
↓
BERT 自己已经训练好的 Embedding

2. BERT采用Padding Mask,因此BERT是双向Encoder，即计算Attention时上下文都能看到

3. token_type_ids 告诉 BERT，一个 token 属于第几个句子，
形式与Padding Mask一样

4. AutoModel:
Tokenizer
    ↓
Embedding
    ↓
BERT Encoder × 12
    ↓
每个 token 的 hidden representation
**只输出Encoder output，不做分类，裸BERT**

AutoModelForSequenceClassification:
             BERT
              ↓
        sentence representation
              ↓
           Linear
              ↓
         5 个 logits
**加上了最后的分类操作**