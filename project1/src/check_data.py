import torch
from pathlib import Path
import pandas as pd

# 1.准备数据
output_path = Path(__file__).parent.parent / "outputs"
output_path.mkdir(parents=True,exist_ok=True) # 如果上级目录不存在，就一起创建。已创建不影响
file_path = Path(__file__).parent.parent / 'data' / 'tickets_1000.csv'
file = pd.read_csv(file_path)

# 2.读取数据概况
print(f'数据尺寸:{file.shape}')
print(f'字段名称:{file.columns.tolist()}')
# print(f'打印前五行:\n{file.head()}')

# 3.检查必要字段
required_columns = [
    "ticket_id",
    "text",
    "label",
    "channel",
    "priority",
    "created_at",
]
miss_columns = []

for col in required_columns:
    if col not in file.columns.tolist():
        miss_columns.append(col)

if len(miss_columns) > 0:
    raise ValueError(f'原始数据的{miss_columns}列缺失')
print('原始数据表不存在列缺失')

# 4.处理缺失值，类型问题和格式问题
file["text"] = file["text"].fillna("") # 缺失值替换为空字符串
file["text"] = file["text"].astype(str) # 确保 text 是字符串
file["text"] = file["text"].str.strip()# 去掉文本前后的空格

# 5.处理缺失值后检查空文本
judge_empty = file['text'] == ''
num_empty = judge_empty.sum()
print("空文本数量：", num_empty)
if num_empty>0:
    print("空文本所在行：")
    print(file.loc[judge_empty,["ticket_id", "text", "label"]]) # 找到为空的行并且打印这些列

# 6.判断重复文本
judge_duplicated = file['text'].duplicated(keep=False) # 把所有重复的文本标记为True
judge_duplicated = judge_duplicated & (file["text"] != "") # 并且是不为空的重复
num_duplicated = judge_duplicated.sum()
print("重复文本行数：", num_duplicated)
if num_duplicated > 0:
    duplicated_data = file.loc[judge_duplicated,["ticket_id", "text", "label"]]
    duplicated_data = duplicated_data.sort_values(by='text')
    print('重复文本：')
    print(duplicated_data)
'''file.loc就是要取哪些行和哪些列'''


# 7. 检查未知标签
valid_labels = {
    "网络故障",
    "费用问题",
    "套餐业务",
    "信号问题",
    "投诉建议",
}
unknown_labels = []

for label in file["label"].dropna().unique():
    if label not in valid_labels:
        unknown_labels.append(label)
print("未知标签：", unknown_labels)

# 8.计算类别
class_distribution = (
    file["label"]
    .value_counts() # 没有列名，只有类别名和个数
    .rename_axis("label") # 命名类别名所在列名为label
    .reset_index(name="count") # 命名个数所在列名为count
) # 得到一个二维csv

valid_label_count = file["label"].notna().sum()
class_distribution['percentage'] = (class_distribution['count'] / valid_label_count * 100).round(2)
# 新增一个percentage列
print("类别分布：")
print(class_distribution)

# 9.统计各个文本的长度
text_len = file['text'].str.len() #file['text']包括索引列和文本列
# print(text_len)
text_length_result = {
    "min": int(text_len.min()),
    "max": int(text_len.max()),
    "mean": round(float(text_len.mean()),2),
    "median": round(float(text_len.median()),2)
}
print("文本长度分布：")
print(text_length_result)


# 12. 整理检查结果
check_result = {
    "total_samples": int(len(file)),
    "empty_text_count": int(num_empty),
    "duplicate_text_count": int(num_duplicated),
    "unknown_labels": unknown_labels,
    "text_length_result":text_length_result,
    "text_length": text_len.tolist(),
}

# 13.保存结果
class_distribution_path = output_path / 'class_distribution.csv'
class_distribution.to_csv(class_distribution_path,index=False,
                          encoding='utf-8')


check_result_path = output_path / 'check_data.json'
import json
with check_result_path.open('w',encoding='utf-8') as json_file:   
    json.dump(
        check_result,
        json_file,
        ensure_ascii=False, # 写中文的格式
        indent=2,# 缩进2格
    )