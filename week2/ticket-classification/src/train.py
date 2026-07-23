import pandas as pd
from common.dataset import TicketDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import TicketClassifier
import torch.nn as nn
import torch
import json
from pathlib import Path


embedding_dim = 64
num_classes = 5
lr = 1e-3
epochs = 10

# 1.读数据
file = pd.read_csv("../data/工单数据集.csv")
texts = file['text'].tolist()
labels = file['label'].tolist()
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,labels,test_size=0.2,random_state=15,stratify=labels
)


# 2.数据存进DataSet
def build_char_to_id(texts):
    char_to_id = {
        '<PAD>':0,
        '<UNK>':1
    }

    for text in texts:
        for char in text:
            if char not in char_to_id:
                char_to_id[char] = len(char_to_id)
    return char_to_id


char_to_id = build_char_to_id(train_texts) # 训练集和测试集都使用一个词表，用的是训练集数据
labels_to_id = {
    "网络故障": 0,
    "费用问题": 1,
    "套餐业务": 2,
    "信号问题": 3,
    "投诉建议": 4,
}
# dataset中存的是每一条数据
'''{
        "text_idx": torch.tensor(text_idx, dtype=torch.long), (max_len)
        "label_idx": torch.tensor(label_idx, dtype=torch.long), (1)
    }'''
train_dataset = TicketDataset(train_texts,train_labels,char_to_id,labels_to_id,max_len=20)
test_dataset = TicketDataset(test_texts,test_labels,char_to_id,labels_to_id,max_len=20)


# 3.构造DataLoader 存的是一批数据(batch_size,max_len)
# 因为每一条padding之后到max_len，DataLoader才能合并每一条数据
train_dataloader = DataLoader(train_dataset,batch_size=8,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=8,shuffle=False)


# 4.模型
model = TicketClassifier(vocab_size=len(char_to_id),embedding_dim=embedding_dim,num_classes=num_classes,
                         padding_idx=char_to_id['<PAD>'])


# 5. 损失函数，优化器
loss_fn = nn.CrossEntropyLoss()
opti = torch.optim.Adam(model.parameters(),lr=lr)


# 6.训练
model.train()

for epoch in range(epochs):

    loss_epoch = 0.0
    num_sample = 0
    for batch in train_dataloader:
        
        opti.zero_grad()

        x = batch['text_idx'] # (batch_size,max_len)编号
        y = batch['label_idx'] # (batch_size)
        #print(y.shape)

        y_pred = model(x)
        loss = loss_fn(y_pred, y) # 这个是每一批次中的平均损失
        loss_epoch += loss.item() * y.size(0)
        num_sample += y.size(0)

        loss.backward()
        opti.step()
    
    avg_loss = loss_epoch / num_sample
    print(
            f"epoch {epoch + 1}/{epochs}, "
            f"loss = {avg_loss:.4f}"
        )


# 7.保存模型
torch.save(model.state_dict(),'../outputs/model.pt') # state_dict是保存字典形式的参数集合，parameters()只是参数

vocab_path = Path('../outputs/vocab.json')
with vocab_path.open('w',encoding='utf-8') as file:   
    json.dump(
        char_to_id,
        file,
        ensure_ascii=False, # 写中文的格式
        indent=2,# 缩进2格
    )


label_to_id_path = Path('../outputs/label_to_id.json')
with label_to_id_path.open('w',encoding='utf-8') as file:
    json.dump(
        labels_to_id,
        file,
        ensure_ascii=False,
        indent=2
    )


print("\n训练完成")
print(f"模型已保存到：outputs/model.pt")
print(f"词表已保存到：{vocab_path}")
print(f"标签映射已保存到：{label_to_id_path}")
