# 以前的csv->python list **一列一列的转化成python list**
# 现在的csv->python list **一段一段的先分隔csv"

# 准备工作设置种子
import random
import numpy
import torch
def set_seed(num_seed:int=15) -> None:
    #1.给python自带的random方法设置种子，伪随机
    random.seed(num_seed)

    # 2.给Numpy的random方法设置种子
    numpy.random.seed(num_seed)

    # 3.给torch的random方法设置种子
    torch.random.manual_seed(num_seed)

    # 给GPU模型参数初始化设置种子，参数初始伪随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num_seed) # 设置当前GPU

def build_char_to_id(texts):
    char_to_id = {
        '<UNK>':0,
        '<PAD>':1
    }

    for text in texts:
        for char in text:
            if char not in char_to_id:
                char_to_id[char] = len(char_to_id)

    return char_to_id

def data_prepare(file_path,train_size,seed,max_len,batch_size):
    
    set_seed(seed)
    
    #1.准备文件
    import pandas as pd
    file = pd.read_csv(file_path)

    #2.划分数据集
    from sklearn.model_selection import train_test_split
    train_file,tmp_file = train_test_split(file,test_size=1-train_size,random_state=seed,
                                            stratify=file['label'])
    valid_file,test_file = train_test_split(tmp_file,test_size=0.5,random_state=seed,
                                            stratify=tmp_file['label'])
    
    #3.准备dataset
    label_to_id = {
        "网络故障": 0,
        "费用问题": 1,
        "套餐业务": 2,
        "信号问题": 3,
        "投诉建议": 4,
    }
    char_to_id = build_char_to_id(train_file['text'].tolist()) # 只用训练集构建词表

    from common.dataset import TicketDataset
    train_dataset = TicketDataset(train_file['text'].tolist(),
                                train_file['label'].tolist(),char_to_id,label_to_id,max_len=max_len)
    valid_dataset = TicketDataset(valid_file['text'].tolist(),
                                valid_file['label'].tolist(),char_to_id,label_to_id,max_len=max_len)
    test_dataset = TicketDataset(test_file['text'].tolist(),
                                test_file['label'].tolist(),char_to_id,label_to_id,max_len=max_len)

    # 3.准备Dataloader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    #4.做必要的检查
    #print(f"训练集的标签类别：\n{train_file['label'].value_counts()}")
    #print(f"验证集的标签类别：\n{valid_file['label'].value_counts()}")
    #print(f"测试集的标签类别：\n{test_file['label'].value_counts()}")

    return {
        "train_dataloader":train_dataloader,
        "valid_dataloader":valid_dataloader,
        "test_dataloader":test_dataloader,
        "char_to_id":char_to_id,
        "label_to_id":label_to_id
    }


if __name__ == "__main__":
    from pathlib import Path

    file_path = Path(__file__).resolve().parent.parent /"data"/"工单数据集.csv"
    data = data_prepare(file_path,train_size=0.3,seed=15,max_len=32,batch_size=8)

