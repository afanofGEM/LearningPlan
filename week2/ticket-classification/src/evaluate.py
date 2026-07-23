import torch

def evaluate(test_dataloader,model,loss_fn):
    # 秒开仙人
    model.eval()

    loss_epoch = 0.0
    num_sample = 0
    num_correct = 0
    with torch.no_grad(): #因为不用更新参数，所以关掉计算过程（计算图）
        for batch in test_dataloader:
            x = batch['text_idx']
            y = batch['label_idx']
            y_pred = model(x) # (batch_size,num_classes)
            loss = loss_fn(y_pred,y)
            loss_epoch += loss.item() * y.size(0)
            num_sample += y.size(0)

            class_pred = y_pred.argmax(dim=1) # (batch_size) 行向量，消除第二维度
            num_correct += (class_pred == y).sum()
        
        avg_loss = loss_epoch / num_sample
        acc = num_correct / num_sample
    return avg_loss,acc


# 1.数据：csv-list/json-dataset-dataloader
import pandas as pd
file = pd.read_csv("../data/工单数据集.csv")
texts = file['text'].tolist()
labels = file['label'].tolist()

from sklearn.model_selection import train_test_split
_,test_texts,_,test_labels = train_test_split(texts,labels,test_size=0.2,
                                              random_state=15,stratify=labels)

from pathlib import Path
import json
vocab_json_path = Path("../outputs/vocab.json")
label_to_id_json_path = Path("../outputs/label_to_id.json")

with vocab_json_path.open('r',encoding='utf-8') as file:
    char_to_id = json.load(file)

with label_to_id_json_path.open('r',encoding='utf-8') as file:
    label_to_id = json.load(file)

# print(char_to_id)
# print(label_to_id)

from common.dataset import TicketDataset
test_dataset = TicketDataset(test_texts,test_labels,char_to_id,label_to_id,max_len=20)

from torch.utils.data import DataLoader
test_dataloader = DataLoader(test_dataset,batch_size=8,shuffle=False)


# 2.model
from model import TicketClassifier
embedding_dim = 64
num_classes = 5
padding_id = 0
model = TicketClassifier(vocab_size=len(char_to_id),embedding_dim=embedding_dim,
                         num_classes=num_classes,padding_idx=padding_id)

'''加载模型：1.从json中接受字典'''
model_path = Path("../outputs/model.pt")
dict = torch.load(model_path)

'''2.用模型接受参数'''
model.load_state_dict(dict)

# 损失函数
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()

# 开始评价
avg_loss,acc = evaluate(test_dataloader,model,loss_fn)
print(f"测试集样本数：{len(test_dataset)}")
print(f"test_loss：{avg_loss:.4f}")
print(f"test_accuracy：{acc:.4f}")
