from pathlib import Path

#1. 准备数据
data_path = Path(__file__).parent.parent / 'data' / 'tickets_1000.csv'
outputs_path = Path(__file__).parent.parent / 'outputs'
outputs_path.mkdir(parents=True,exist_ok=True)
train_path = Path(__file__).parent.parent / 'data' / 'train.csv'
eval_path = Path(__file__).parent.parent / 'data' / 'eval.csv'
test_path = Path(__file__).parent.parent / 'data' / 'test.csv'

import pandas as pd
def prepare_data(train_path,eval_path,test_path):
    train_file = pd.read_csv(train_path)
    eval_file = pd.read_csv(eval_path)
    test_file = pd.read_csv(test_path)

    train_texts = train_file["text"].tolist()
    train_labels = train_file["label"].tolist()
    eval_texts = eval_file["text"].tolist()
    eval_labels = eval_file["label"].tolist()
    test_texts = test_file['text'].tolist()
    test_labels = test_file['label'].tolist()

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "eval_texts": eval_texts,
        "eval_labels": eval_labels,
        "test_texts":test_texts,
        "test_labels":test_labels
    }


# 2.构建TF-IDF特征：
def build_tfidf(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer='char',ngram_range=(1,2))

    # 对训练集数据编码
    train_features = tfidf.fit_transform(data['train_texts'])

    # 用训练集的词典对验证集编码
    eval_features = tfidf.transform(data['eval_texts'])

    print("\n训练集 TF-IDF 形状：",train_features.shape)
    print("验证集 TF-IDF 形状：",eval_features.shape)

    return {
        "tfidf": tfidf,
        "train_features": train_features,
        "eval_features": eval_features
    }


# 3.评价模型
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
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
        class_result = report[c]
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


# 4.训练过程,输入的data而不是路径
def run_experiment(data,c_value,max_iter,random_state):
    dataset = build_tfidf(data)
    '''{
        "tfidf": tfidf,
        "train_features": train_features,
        "eval_features": eval_features
    }data经历了从list到编码list'''

    # 这是全部的训练集数据
    train_data = dataset['train_features']
    eval_data = dataset['eval_features']

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=c_value,max_iter = max_iter,random_state=random_state) 
    '''C控制模型对复杂程度的限制，也就是正则化强度的倒数
        max_iter表示训练的轮数'''

    model.fit(train_data,data['train_labels'])
    print('模型训练完成')

    eval_predictions = model.predict(eval_data)
    metrics = evaluate(data['eval_labels'],eval_predictions)

    return {
        'tfidf':dataset['tfidf'],
        'model':model,
        'metrics':metrics,
        'model_conf':{
            'c-value':c_value,
            'max_iter':max_iter,
            'random_state':random_state,
        },
        'tfidf_conf':{
            'analyzer':'char',
            'ngram_range':(1,2)
        }
    }


def save_results(results):

    # 1. 保存模型和TF-IDF
    import joblib
    model_path = outputs_path / 'model.joblib'
    model = results['model']
    joblib.dump(model, model_path)

    tfidf_path = outputs_path / "tfidf.joblib"
    joblib.dump(results["tfidf"], tfidf_path)


    # 2.保存模型参数
    import json
    model_conf_path = outputs_path / 'model_conf.json'
    model_conf = results['model_conf']
    with model_conf_path.open('w',encoding='utf-8') as model_file:   
         json.dump(
            model_conf,
            model_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )

    # 3.保存验证集指标
    metrics_path = outputs_path / 'metrics.json'
    metrics = results['metrics']
    with metrics_path.open('w',encoding='utf-8') as metrics_file:   
         json.dump(
            metrics,
            metrics_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )

    # 4.保存tfidf:
    tfidf_conf_path = outputs_path / 'tfidf_conf.json'
    tfidf = results['tfidf_conf']
    with tfidf_conf_path.open('w',encoding='utf-8') as tfidf_file:   
         json.dump(
            tfidf,
            tfidf_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )


def main():
    data = prepare_data(train_path,eval_path,test_path)
    '''{
        "train_texts": train_texts,
        "train_labels": train_labels,
        "eval_texts": eval_texts,
        "eval_labels": eval_labels,
        "test_texts":test_texts,
        "test_labels":test_labels
    }'''

    # dataset = build_tfidf(data)
    '''{
        "tfidf": tfidf,
        "train_features": train_features,
        "eval_features": eval_features
    }'''

    random_state = 15
    max_iter = 1000
    c_value = 1.0
    results = run_experiment(data,c_value,max_iter,random_state) #内部会转成dataset
    '''{
        'tfidf':dataset['tfidf'],
        'model':model,
        'metrics':metrics,
        'model_conf':{
            'c-value':c_value,
            'max_iter':max_iter,
            'random_state':random_state,
        },
        'tfidf_conf':{
            'analyzer':'char',
            'ngram_range':(1,2)
        }
    }'''

    save_results(results)


if __name__ == "__main__":
    main()
    print('done')