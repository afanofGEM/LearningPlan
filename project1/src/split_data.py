from sklearn.model_selection import train_test_split
from pathlib import Path

# 1.设置路径
data_path = Path(__file__).parent.parent / 'data' / 'tickets_1000.csv'
csv_path = Path(__file__).parent.parent / 'data'
outputs_path = Path(__file__).parent.parent / 'outputs'
outputs_path.mkdir(parents=True,exist_ok=True)
train_csv_path = csv_path / 'train.csv'
eval_csv_path = csv_path / 'eval.csv'
test_csv_path = csv_path / 'test.csv'
train_distribution_csv_path = outputs_path / 'train_distribution.csv'
eval_distribution_csv_path = outputs_path / 'eval_distribution.csv'
test_distribution_csv_path = outputs_path / 'test_distribution.csv'

# 2.准备数据集
test_size = 0.3
seed = 15
import pandas as pd
dataframe = pd.read_csv(data_path)
train_dataframe,tmp_dataframe = train_test_split(dataframe,test_size=test_size,
                                   random_state=seed,shuffle=True,
                                   stratify=dataframe['label']) # 保留类别比例不变
eval_dataframe,test_dataframe = train_test_split(tmp_dataframe,test_size=0.5,
                                                 random_state=seed,shuffle=True,
                                                 stratify=tmp_dataframe['label'])

'''重置索引'''
train_dataframe = train_dataframe.reset_index(drop=True)
eval_dataframe = eval_dataframe.reset_index(drop=True)
test_dataframe = test_dataframe.reset_index(drop=True) 
# 因为随机划分会打乱数据的索引，所以重新创建csv时需要重新划分索引

# 3. 简单的结果统计
count_train = train_dataframe['label'].value_counts().rename_axis('label').reset_index(name='count')
count_train['per'] = (count_train['count'] / train_dataframe['label'].notna().sum() * 100).round(2)

count_eval = eval_dataframe['label'].value_counts().rename_axis('label').reset_index(name='count')
count_eval['per'] = (count_eval['count'] / eval_dataframe['label'].notna().sum() * 100).round(2)

count_test = test_dataframe['label'].value_counts().rename_axis('label').reset_index(name='count')
count_test['per'] = (count_test['count'] / test_dataframe['label'].notna().sum() * 100).round(2)

# 4. 保存结果
count_train.to_csv(train_distribution_csv_path,index=False,encoding='utf-8')
count_eval.to_csv(eval_distribution_csv_path,index=False,encoding='utf-8')
count_test.to_csv(test_distribution_csv_path,index=False,encoding='utf-8')

train_dataframe.to_csv(train_csv_path,index=False,encoding='utf-8')
eval_dataframe.to_csv(eval_csv_path,index=False,encoding='utf-8')
test_dataframe.to_csv(test_csv_path,index=False,encoding='utf-8')