# 客服工单文本分类

## 1. 项目背景

本项目实现了一个简单的中文客服工单分类模型。

用户输入一段客服工单文本，模型根据文本内容预测工单所属类别。

例如：

```text
输入：我家宽带一直连不上
输出：网络故障
```

本项目主要用于学习 PyTorch **中文文本分类的完整流程**，包括：

* 构造文本分类数据集
* 建立字符词表
* 将中文文本转换为 input_ids
* 使用 Dataset 和 DataLoader 组织数据
* 构建简单神经网络模型
* 训练与评估模型
* 保存和加载模型
* 对新文本进行预测

---

## 2. 数据类别

当前项目包含五个工单类别：

| 类别   | 含义                     |
| ---- | ---------------------- |
| 网络故障 | 家庭宽带、Wi-Fi、路由器、光猫等网络问题 |
| 费用问题 | 话费、账单、扣费、欠费和充值问题       |
| 套餐业务 | 套餐办理、取消、更换和流量包问题       |
| 信号问题 | 手机信号、4G、5G和通话质量问题      |
| 投诉建议 | 服务投诉、维修投诉和业务建议         |

---

## 3. 数据格式

数据保存在：

```text
data/工单数据集.csv
```

CSV 文件包含两列：

```text
text,label
```

示例：

```csv
text,label
我家宽带突然断网了,网络故障
这个月话费为什么多扣了,费用问题
我想换一个流量更多的套餐,套餐业务
我在宿舍里手机信号很差,信号问题
我要投诉维修师傅一直没来,投诉建议
```

---

## 4. Dataset 和 DataLoader

### Dataset

Dataset 负责处理单条样本。

它将原始文本：

```text
我家宽带断网了
```

转换为固定长度的字符编号：

```text
[12, 8, 25, 16, 31, 7, 4, 0, 0, ...]

return {
    "text_idx": torch.tensor(text_idx, dtype=torch.long),
    "label_idx": torch.tensor(label_idx, dtype=torch.long)
}
```

同时将文字标签转换为数字标签：**主要靠encode_text方法，同时注意截断和padding**

```text
网络故障 → 0
```

一条样本最终返回：

```python
{
    "text_idx": tensor([...]), # 这里的长度都是max_len
    "label_idx": tensor(0),
}
```

### DataLoader

DataLoader 负责将多条样本组成 batch。

例如：

```text
batch_size = 8
max_len = 20
```

一个 batch 的 shape 为：

```text
input_ids：[8, 20]
labels：[8]
```

训练集使用：

```python
shuffle=True
```

测试集使用：

```python
shuffle=False
```

---

## 5. 模型结构

当前模型结构为：

```text
input_ids
→ Embedding
→ masked mean pooling
→ Linear
→ logits
```

假设：

```text
batch_size = 8
max_len = 20
embedding_dim = 64
num_classes = 5
```

数据 shape 变化为：

```text
input_ids：[8, 20]
Embedding 后：[8, 20, 64]
mean pooling 后：[8, 64]，对于每一条[20,64]文本，按照dim=1(20)维度方向求平均值(/20) 
Linear 后：[8, 5]
```

最终的 `[8, 5]` 表示：

```text
当前 batch 有 8 条文本
每条文本对应 5 个类别分数
```

---

## 6. 项目结构

```text
ticket-classification/
├── data/
│   └── 工单数据集.csv
├── outputs/
│   ├── model.pt
│   ├── vocab.json
│   ├── label_to_id.json
│   └── config.json
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── README.md
└── requirements.txt
```

---

## 7. 如何训练

在项目根目录运行：

```bash
python src/train.py
```

训练过程中会输出：

```text
epoch 1/10, train_loss=1.6182, test_loss=1.6075, test_acc=0.2667
epoch 2/10, train_loss=1.5601, test_loss=1.5732, test_acc=0.3333
```

训练完成后会保存：

```text
outputs/model.pt
outputs/vocab.json
outputs/label_to_id.json
outputs/config.json
```

这些文件分别用于保存：

* `model.pt`：模型参数
* `vocab.json`：字符与编号的映射
* `label_to_id.json`：类别与编号的映射
* `config.json`：模型和数据处理配置

---

## 8. 如何评估

在项目根目录运行：

```bash
python src/evaluate.py
```

程序会在测试集上输出：

```text
测试集样本数：15
test_loss：1.2356
test_accuracy：0.6667
```

评估阶段使用：

```python
model.eval()
```

和：

```python
with torch.no_grad():
```

评估过程中不会更新模型参数。

---

## 9. 如何预测

在项目根目录运行：

```bash
python src/predict.py
```

示例输出：

```text
输入：我家宽带一直连不上
预测类别：网络故障

输入：这个月话费怎么比上个月贵
预测类别：费用问题

输入：我想换一个流量更多的套餐
预测类别：套餐业务
```

预测时必须加载训练阶段保存的词表，不能重新建立 vocab。

---

## 10. 当前不足

目前使用的是 toy 数据集，样本数量较少，文本表达方式比较简单，因此当前模型的测试结果不能代表真实客服业务中的实际效果。

当前模型只是：

```text
Embedding
+ mean pooling
+ Linear
```

模型没有考虑复杂的词序关系，也没有使用 CNN、RNN、Transformer 或 BERT。

当前项目的主要目标是理解 PyTorch 文本分类的完整工程流程，而不是训练生产级模型。

后续计划：

* Week 3：加入 TF-IDF 和传统机器学习 baseline
* 后续阶段：学习 CNN、RNN 和 Transformer
* Week 5-6：使用 HuggingFace 和预训练 BERT 模型
* 增加更大规模、更真实的客服工单数据
* 增加 precision、recall、F1-score 和混淆矩阵

---

## 12. 本周总结

本周完成了一个最小可运行的中文文本分类项目。

完整流程为：

```text
原始 CSV
→ train/test split
→ 构建字符词表
→ 文本转换为 input_ids (encode_text同时截断和padding)
→ Dataset (对每一条数据格式化，融合了上一条的功能)
→ DataLoader
→ Embedding
→ mean pooling
→ Linear
→ CrossEntropyLoss
→ 模型训练
→ 测试集评估
→ 单句预测
```

通过本项目，理解了以下核心概念：

* 文本为什么需要转换为数字
* Dataset 如何处理单条样本
* DataLoader 如何组织 batch
* Embedding 如何将字符编号转换为向量
* mean pooling 如何生成文本表示
* logits、loss 和类别预测之间的关系
* `model.train()` 与 `model.eval()` 的区别
* 模型参数、词表和标签映射为什么需要一起保存
