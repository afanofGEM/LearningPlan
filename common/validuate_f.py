from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def evaluate(labels,predictions):

    # 1.计算整体指标
    accuracy = accuracy_score(labels,predictions)
    precision_macro = precision_score(labels,predictions,average="macro",zero_division=0)
    recall_macro = recall_score(labels,predictions,average="macro",zero_division=0)
    f1_macro = f1_score(labels,predictions,average="macro",zero_division=0)

    '''它会计算每个类别的：
        precision：预测成这个类别的样本中，有多少是真的
        recall：真实属于这个类别的样本中，有多少被找出来了
        f1-score：Precision 和 Recall 的综合指标
        support：该类别真实样本数量'''

    # 2. 得到每个类别的指标
    '''{
        "网络故障": {
            "precision": 1.0,
            "recall": 0.5,
            "f1-score": 0.67,
            "support": 2.0,
        },
        "费用问题": {
            "precision": 0.67,
            "recall": 1.0,
            "f1-score": 0.8,
            "support": 2.0,
        },
        "accuracy": 0.75,
        "macro avg": {
            "precision": 0.83,
            "recall": 0.75,
            "f1-score": 0.73,
            "support": 4.0,
            },
        "weighted avg": {
            "precision": 0.83,
            "recall": 0.75,
            "f1-score": 0.73,
            "support": 4.0,
        },
       }'''
    report = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    '''做一遍搬运，只留下每个类别的值'''
    class_metrics = {}
    class_name = sorted(set(labels))
    for c in class_name:
        class_result = report[str(c)]
        class_metrics[c] = {
            "precision": round(float(class_result["precision"]),4,),
            "recall": round(float(class_result["recall"]),4),
            "f1-score": round(float(class_result["f1-score"]),4),
            "support": int(class_result["support"]), # 样本数
        }

    # 3. 整理完整评价结果
    metrics = {
        "accuracy": round(float(accuracy),4),
        "precision_macro": round(float(precision_macro),4),
        "recall_macro": round(float(recall_macro),4),
        "f1_macro": round(float(f1_macro),4),
        "per_class": class_metrics
    }

    return metrics


def plot_confusion_matrix(labels, predictions, class_names, save_path):
    # labels/predictions 使用整数类别 ID
    class_ids = list(range(len(class_names)))

    matrix = confusion_matrix(
        y_true=labels,
        y_pred=predictions,
        labels=class_ids
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(values_format="d", xticks_rotation=30)

    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("测试集混淆矩阵")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()