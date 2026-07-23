from .models.mlp import classifier_mlp
from .data_prepare import data_prepare
import pandas as pd
from pathlib import Path
import torch

train_size = 0.6
seed = 15
max_len = 32
batch_size = 8

data_path = Path(__file__).parent.parent / "data" / "工单数据集.csv"
data = data_prepare(data_path,train_size=train_size,seed=seed,max_len=max_len,batch_size=batch_size)

def collect_pred(testdataloader,model):
    model.eval()

    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch in testdataloader:
            x = batch['text_idx']
            y = batch['label_idx']
            y_pred = model(x) # （batch_size,class_num)
            y_pred_probability = torch.softmax(y_pred,dim=1) # 对每行的数据进行softmax
            probability,class_pred = torch.max(y_pred_probability,dim=1) # (batch_size,)，表示行向量
            all_pred.extend(class_pred.tolist())
            all_label.extend(y.tolist())

    return all_pred,all_label


import matplotlib.pyplot as plt
# 让 Matplotlib 正确显示中文
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# 避免坐标轴负号显示异常
plt.rcParams["axes.unicode_minus"] = False

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def calculate_metrics(labels,predictions,save_path):

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels,predictions,average="macro",
                                                zero_division=0)),
        "recall_macro": float(recall_score(labels,predictions,average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(labels,predictions,average="macro",zero_division=0)),
    }

    import json
    with save_path.open('w',encoding='utf-8') as file:   
         json.dump(
            metrics,
            file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )
    return metrics


def plot_loss_curve(history,save_path):
    epochs = range(1,len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs,history["train_loss"],label="Train Loss")

    plt.plot(epochs,history["eval_loss"],label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path,dpi=300)

    plt.close()


def plot_confusion_matrix(labels,predictions,label_to_id,save_path):
    label_id = range(0,len(label_to_id))

    # label_id决定了混淆矩阵的尺寸
    '''行：真实类别   列：预测类别
       例如第一行：[2, 0, 0, 1, 0]
       表示真实类别为 0，也就是“网络故障”的样本中：
        2 条预测成网络故障
        0 条预测成费用问题
        0 条预测成套餐业务
        1 条预测成信号问题
        0 条预测成投诉建议'''
    matrix = confusion_matrix(y_true=labels,y_pred=predictions,labels=label_id)

    '''label_to_id.items()得到：
        [
            ("网络故障", 0),
            ("费用问题", 1),
            ("套餐业务", 2),
            ("信号问题", 3),
            ("投诉建议", 4),
        ]这样才能用item[1]排序'''
    class_name = [label for label,idx in sorted(label_to_id.items(),
                                                key= lambda item:item[1])]

    # 把坐标轴序号变成名称
    display = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=class_name)

    #values_format="d"表示矩阵中的数字按照整数显示
    #xticks_rotation=30表示横轴类别名称旋转 30 度
    display.plot(values_format="d",xticks_rotation=30)

    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("测试集混淆矩阵")

    '''它会自动调整图像边距，避免：
    标题被裁掉
    坐标轴名称被裁掉
    类别名称显示不完整'''
    plt.tight_layout()

    plt.savefig(save_path,dpi=300) #表示使用较高分辨率保存，图片会更清晰
    plt.close()


def main():

    # 1.加载模型,加载参数以初始化
    best_model_config_path = Path(__file__).parent.parent / "outputs" / "best_model_config.json"

    import json
    with best_model_config_path.open('r',encoding='utf-8') as file:
        config = json.load(file)

    model = classifier_mlp(vocab_size=len(data['char_to_id']),
                           embedding_dim=config['embedding_dim'],
                           padding_id=data['char_to_id']['<PAD>'],
                           hidden_dim=config['hidden_dim'],
                           num_classes=len(data['label_to_id']),
                           dropout=config['dropout'])

    # 加载模型参数
    model_path = Path(__file__).parent.parent / "outputs" / "best_model.pt"
    param_dict = torch.load(model_path)
    '''用模型接受参数'''
    model.load_state_dict(param_dict)

    predictions,labels = collect_pred(data['test_dataloader'],model)

    # 2.计算保存测试集指标
    metrics_save_path = Path(__file__).parent.parent / "outputs" / "metrics.json"
    metrics = calculate_metrics(labels,predictions,metrics_save_path)

    print("测试集指标：")
    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    # 3.画混淆矩阵
    confusion_matrix_save_path = Path(__file__).parent.parent / "outputs" / "confusion_matrix.png"
    plot_confusion_matrix(labels,predictions,data['label_to_id'],
                          save_path=confusion_matrix_save_path)

    # 4.画history
    history_path = Path(__file__).parent.parent / "outputs" / "history.json"
    history_save_path = Path(__file__).parent.parent / "outputs" / "history.png"

    with history_path.open('r',encoding='utf-8') as file:
        history = json.load(file)

    plot_loss_curve(history,save_path=history_save_path)
    

if __name__ == "__main__":
    main()