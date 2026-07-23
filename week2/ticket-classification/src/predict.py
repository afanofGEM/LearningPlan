import torch


def predict(text,model,id_to_label): # 编码好的文本(max_len)
    text = torch.tensor(text,dtype=torch.long)
    text = text.unsqueeze(0) #(max_len)-(1,max_len)增加第一维度

    model.eval()
    with torch.no_grad():
        y_pred = model(text) # (1,num_classes)
    
    class_pred = y_pred.argmax(dim=1) #(1) tensor([0])需要用item()读数值
    # print(class_pred)
    return id_to_label[class_pred.item()]


# 1.预测文本
test_texts = [
    "我家宽带一直连不上",
    "这个月话费怎么比上个月贵",
    "我想换一个流量更多的套餐",
    "我这里手机信号特别差",
    "我要投诉维修一直没人处理",
]


# 2.文本编码函数
def encode_text(text,char_to_id,max_len=20):
    '''先截断'''
    text = text[:max_len]

    ids = []
    for char in text:
        if char not in char_to_id:
            ids.append(char_to_id['<UNK>'])
        else:
            ids.append(char_to_id[char])
    
    while len(ids) < max_len:
        ids.append(char_to_id['<PAD>'])
    
    return ids


# 3.加载json和模型
import json
from pathlib import Path
vocab_json_path = Path("../outputs/vocab.json")
label_to_id_json_path = Path("../outputs/label_to_id.json")

with vocab_json_path.open('r',encoding='utf-8') as file:
    char_to_id = json.load(file)

with label_to_id_json_path.open('r',encoding='utf-8') as file:
    label_to_id = json.load(file)


id_to_label = {id:label for label,id in label_to_id.items()} #items读取键值对
# print(id_to_label)

from model import TicketClassifier
vocab_size = len(char_to_id)
embedding_dim = 64
num_classes = 5
padding_id = 0
model = TicketClassifier(vocab_size,embedding_dim,num_classes,padding_id)
model_path = Path("../outputs/model.pt")
dict = torch.load(model_path)
'''用模型接受参数'''
model.load_state_dict(dict)


# 4.开始预测
list_pred = []
for text in test_texts:
    text_ids = encode_text(text,char_to_id,max_len=20)
    class_pred = predict(text_ids,model,id_to_label=id_to_label)
    print(f"输入的文本:{text}")
    print(f"属于的问题类别:{class_pred}")


    