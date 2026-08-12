# 得开代理运行
from transformers import AutoTokenizer

MODEL_NAME = "google-bert/bert-base-chinese"

texts = [
    "手机没有信号",
    "宽带突然断网了",
    "这个月的话费为什么这么高",
    "想把现在的套餐改成便宜一点的",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encoded = tokenizer(
    texts,                 # 把这些文本进行编码
    padding=True,          # 短句补 PAD
    truncation=True,       # 长句截断
    return_tensors="pt",   # 转成 PyTorch Tensor
)
'''{
    'input_ids':(batch_size,text_max_len) 是文本中的最长句长度
    'attention_mask':(batch_size,text_max_len)
    'token_type_ids':(batch_size,text_max_len)表示每个token在第几句话的
}'''


print("Tokenizer 输出")
print("\ninput_ids:")
print(encoded["input_ids"])

print("\nattention_mask:")
print(encoded["attention_mask"])

if "token_type_ids" in encoded:
    print("\ntoken_type_ids:")
    print(encoded["token_type_ids"])

print("\ninput_ids shape:")
print(encoded["input_ids"].shape)

print("\nattention_mask shape:")
print(encoded["attention_mask"].shape)

print("逐条查看")

for index, text in enumerate(texts): #enumerate就会多一个索引值

    '''每条文本的id'''
    input_ids = encoded["input_ids"][index]

    '''每条文本的所有Token'''
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    '''根据id还原文本'''
    decoded_text = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
    )

    print(f"\n原始文本：{text}")
    print(f"tokens：{tokens}")
    print(f"input_ids：{input_ids}")
    print(f"attention_mask：{encoded['attention_mask'][index]}")

    if "token_type_ids" in encoded:
        print(f"token_type_ids：{encoded['token_type_ids'][index]}")

    print(f"decode：{decoded_text}")