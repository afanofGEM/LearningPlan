# 客服工单文本分类项目

## 1. 项目背景

本项目用于实现电信客服工单的自动分类。

模型接收一段客服工单文本，并预测该工单所属的业务类别，从而辅助后续的工单分派和处理。

本项目首先使用 TF-IDF 和 Logistic Regression 建立 baseline，后续将进一步尝试 CNN、RNN、LSTM 和 Transformer 等模型。

## 2. 数据结构

数据集包含以下字段：

| 字段         | 说明     |
| ---------- | ------ |
| ticket_id  | 工单唯一编号 |
| text       | 工单文本   |
| label      | 工单类别   |
| channel    | 工单来源渠道 |
| priority   | 工单优先级  |
| created_at | 工单创建时间 |

模型当前只使用 `text` 作为输入，使用 `label` 作为预测目标。

## 3. 标签体系

项目包含五个类别：

* 网络故障
* 费用问题
* 套餐业务
* 信号问题
* 投诉建议

其中，网络故障主要表示宽带、Wi-Fi 和上网异常；信号问题主要表示移动网络、通话信号和区域覆盖异常。

## 4. 数据划分

数据按照以下比例进行分层划分：

* 训练集：70%
* 验证集：15%
* 测试集：15%

划分时使用：

```python
random_state=42
stratify=dataframe["label"]
```

从而保证实验可以复现，并尽量保持三个数据集中的类别比例一致。

## 5. Baseline 模型

Baseline 使用：

```text
字符级 TF-IDF + Logistic Regression
```

TF-IDF 参数：

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 2),
)
```

该配置同时提取单字符和双字符特征。

## 6. 运行方式

首先进行数据检查：

```bash
python -m src.data.check_data
```

然后划分数据：

```bash
python -m src.data.split_data
```

训练 baseline：

```bash
python -m src.baselines.train_tfidf_lr
```

最后在测试集上进行评价：

```bash
python -m src.baselines.evaluate_tfidf_lr
```

## 7. 评价指标

项目使用以下指标：

* Accuracy
* Macro-F1 (此标准用来选择最佳模型)
* 每个类别的 Precision
* 每个类别的 Recall
* 每个类别的 F1

同时生成混淆矩阵和错误案例文件。

## 8. 当前结果

验证集结果：

* Accuracy：1.0
* Macro-F1：1.0

测试集结果：

* Accuracy：1.0
* Macro-F1：1.0

结果文件保存在 `outputs` 文件夹中。

## 9. 错误案例分析

当前重点分析以下类别之间的混淆：

* 网络故障与信号问题
* 套餐业务与费用问题
* 投诉建议与具体业务类别

可能的错误原因包括：

* 标签边界不够清晰
* 一条文本包含多个意图
* 模拟数据的表达方式比较单一
* 某些类别的高频关键词发生重叠
* TF-IDF 无法充分理解否定、语序和上下文

错误案例保存在：

```text
outputs/error_cases.csv
```

## 10. 当前局限

当前数据主要为模拟数据，与真实客服工单仍存在差异。

模拟数据可能缺少：

* 错别字
* 方言和口语缩写
* 极端短文本
* 多意图文本
* 上下文缺失
* 类别分布不平衡

因此，模拟数据上的高准确率不能直接代表模型在真实业务中的表现。

## 11. 下一周计划

下一周计划包括：

* 建立字符级 CNN 文本分类模型
* 建立 RNN 或 LSTM 文本分类模型
* 对比传统 baseline 与神经网络模型
* 分析不同模型的错误案例
* 完善项目文档与实验记录
