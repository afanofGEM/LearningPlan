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
def run_experiment(data,max_iter,random_state):
    '''data:
    {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "eval_texts": eval_texts,
        "eval_labels": eval_labels,
        "test_texts":test_texts,
        "test_labels":test_labels
    }'''
    # GridSearchCV 参数网络 搜索 交叉验证：通过训练和验证集来搜索参数与超参数的工具

    # 1.GridSearCV的数据准备部分
    # 1.1 GridSearchCV需要完整的训练集+验证集
    search_texts = data['train_texts'] + data['eval_texts']
    search_labels = data['train_labels'] + data['eval_labels']

    # 1.2 GridSearchCV需要通过索引告诉它哪些是训练集，哪些是验证集
    index = [-1] * len(data['train_texts']) + [0] * len(data['eval_texts'])
    from sklearn.model_selection import GridSearchCV,PredefinedSplit
    predefined_split = PredefinedSplit(index)

    # 2.GridSearchCV的工作流定义部分
    # 2.1 创建Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pipeline = Pipeline(steps=[('tfidf',TfidfVectorizer(analyzer='char',
                                                        ngram_range=(1,2))),
                               ('model',LogisticRegression(
                                                            max_iter=max_iter,
                                                            solver="lbfgs",
                                                            random_state=random_state))])
    '''lbfgs 会不断寻找让损失函数更小的参数，是 scikit-learn 逻辑回归的默认求解器'''

    # 3.GridSearchCV的参数范围
    # 因为LogisticRegression位于Pipeline的model步骤中，所以参数名称需要写成model__参数名
    params_range = {
        'model__C':[0.01,0.1,1,10,100],
        'model__class_weight':[None,'balanced']
    }

    # 4.创建GridSearchCV
    grid_search_cv = GridSearchCV(estimator=pipeline,
                               param_grid=params_range,# 使用宏平均F1选择最佳参数
                               scoring="f1_macro",cv=predefined_split,n_jobs=-1,
                               # 选择最佳参数后,不使用训练集 + 验证集重新训练最终模型
                               refit=False)
    
    # 5.开始寻找参数
    grid_search_cv.fit(X=search_texts,y=search_labels)

    # 6.得到最优参数
    best_c = grid_search_cv.best_params_["model__C"]
    best_class_weight = grid_search_cv.best_params_["model__class_weight"]
    best_f1_score = grid_search_cv.best_score_

    # 7.使用最优超参数再次创建模型，进行一次train+eval训练
    best_pipeline = Pipeline(steps=[('tfidf',TfidfVectorizer(analyzer='char',
                                                        ngram_range=(1,2))),
                               ('model',LogisticRegression(
                                                            max_iter=max_iter,
                                                            solver="lbfgs",
                                                            random_state=random_state,
                                                            C=best_c,
                                                            class_weight=best_class_weight))])

    '''因为确定了超参数，所以不需要GridSearchCV'''
    best_pipeline.fit(X=search_texts,y=search_labels)

    # 10.取模型和TF-IDF
    best_tfidf = best_pipeline.named_steps["tfidf"]
    best_model = best_pipeline.named_steps["model"]

    return {
        'tfidf':best_tfidf,
        'model':best_model,
        'metrics':
        {
            'best_eval_f1_score':best_f1_score
        },
        'model_conf':{
            'c-value':best_c,
            'class_weight':best_class_weight,
            'max_iter':max_iter,
            'solver':"lbfgs",
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
    eval_metrics_path = outputs_path / 'eval_metrics.json'
    metrics = results['metrics']
    with eval_metrics_path.open('w',encoding='utf-8') as eval_metrics_file:   
         json.dump(
            metrics,
            eval_metrics_file,
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

    random_state = 15
    max_iter = 1000
    results = run_experiment(data,max_iter,random_state) #内部会转成dataset
    '''{
        'tfidf':dataset['tfidf'],
        'model':model,
        'metrics':metrics,
        'model_conf':{
            'c_value':c_value,
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