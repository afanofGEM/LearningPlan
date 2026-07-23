from common.dataset import TicketDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import Counter

file = pd.read_csv("../data/工单数据集.csv")
texts = file["text"].tolist()
labels = file["label"].tolist()

# 1.从csv划分数据集列表
'''stratify表示按照类别划分
优先保证测试集的样本数'''
train_texts,test_texts,train_labels,test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42,stratify=labels) # 并不改变尺寸

print(f"训练集的标签分布是{Counter(train_labels)}")
print(f"测试集的标签分布是{Counter(test_labels)}")

# 2.从数据集列表构建词表
def build_char_to_idx(texts):
    char_to_idx = {"<PAD>": 0, "<UNK>": 1} # padding和unknown

    for text in texts:
        for char in text:
            if char not in char_to_idx:
                char_to_idx["char"] = len(char_to_idx)
    
    return char_to_idx


labels_to_idx = {
    "网络故障": 0,
    "费用问题": 1,
    "套餐业务": 2,
    "信号问题": 3,
    "投诉建议": 4,
}
char_to_idx = build_char_to_idx(train_texts)

# 注意训练集和测试集必须使用同一个 vocab,否则同一个字在两个词表中的编号可能不同
train_dataset = TicketDataset(train_texts,train_labels,char_to_idx,
                                  labels_to_idx,max_len=32)
test_dataset = TicketDataset(test_texts,test_labels,char_to_idx,
                                 labels_to_idx,max_len=32)

train_dataloader = DataLoader(train_dataset,batch_size=8,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=8,shuffle=False)


print(f"一共有{len(texts)}个样本")
print(f"训练集一共有{len(train_dataset)}个样本")
print(f"测试集一共有{len(test_dataset)}个样本")
print(f"词表的大小是{len(char_to_idx)}")
print(f"训练集一共有{len(train_dataloader)}个批次")
print(f"测试集一共有{len(test_dataloader)}个批次")

for batch in train_dataloader:
    print(f"每个batch中输入数据的尺寸为{batch['label_idx'].shape}") # 这里键的名称与DataSet一样
    print(batch['text_idx'])
    print(f"每个batch中标签的尺寸为{batch['label_idx'].shape}")
    print(batch['label_idx'])

