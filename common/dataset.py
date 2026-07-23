'''Dataset 的职责是:给我一个下标，我返回一条数据。'''

from torch.utils.data import Dataset
import torch
import pandas as pd

'''texts的形式是一个列表，元素是每一个输入的文本
   通过csv文件读取data label列转化为数据即可'''
def build_char_to_idx(texts):
    '''texts = 
    [
    "宽带断网",
    "手机信号差"
    ]'''
    char_to_idx = {
        "<UNK>": 0,  # unknown
        "<PAD>": 1,  # padding"
    }

    for text in texts:
        for char in text:
            if char not in char_to_idx:
                char_to_idx[char] = len(char_to_idx) # 妙！现有的字典长度就是下一个索引
    return char_to_idx


# text是列表中的一个元素  "我家宽带有问题"
def encode_text(text, char_to_idx,max_len):
    # max_len是为了保证每个文本的长度一致，方便后续的批处理
    text = text[:max_len]  # 截断 "我家宽带一直连接不上"，如果长度不到max的话不改变len
    idx_list = []

    for char in text:
        if char in char_to_idx:
            idx_list.append(char_to_idx[char])
        else:
            idx_list.append(char_to_idx["<UNK>"])  # unknown
    
    # padding
    '''padding的作用是：只有相同长度的单条数据才能被DataLoader合并'''
    while len(idx_list) < max_len:
        idx_list.append(char_to_idx["<PAD>"])  # padding

    return idx_list


class TicketDataset(Dataset):
    def __init__(self, texts, labels, char_to_idx, label_to_idx,max_len=32):
        
        '''texts是与上文格式相同的数据，labels是真实标签，两个词表以及规定输入长度'''
        self.texts = texts
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.label_to_idx = label_to_idx
        self.max_len = max_len
    

    def __len__(self): # 得到数据的长度
        return len(self.texts)


    '''实现核心功能：“我家宽带有问题”，“网络问题” 
                    变成：
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]，长度为max_len
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]，长度为max_len
    '''
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_idx = encode_text(text,self.char_to_idx,self.max_len)
        label_idx = self.label_to_idx[label]
        return {
            "text_idx": torch.tensor(text_idx, dtype=torch.long),
            "label_idx": torch.tensor(label_idx, dtype=torch.long)
        }


def test():

    file = pd.read_csv("../data/工单数据集.csv")
    texts = file['text'].tolist()
    labels = file['label'].tolist()
    char_to_idx = build_char_to_idx(texts)
    labels_to_idx = {
        "网络故障": 0,
        "费用问题": 1,
        "套餐业务": 2,
        "信号问题": 3,
        "投诉建议": 4,
    }

    dataset = TicketDataset(texts, labels, char_to_idx, labels_to_idx, max_len=32)
    data = dataset[0]
    print(f"样本数: {len(dataset)}")
    print(f"词表大小: {len(char_to_idx)}")
    print(f"原始文本: {texts[0]}")
    print(f"编码后的文本: {data['text_idx']}")
    print(f"原始标签: {labels[0]}")
    print(f"编码后的标签: {data['label_idx']}")


if __name__ == "__main__":
    test()
