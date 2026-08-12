1. 常用库：
   1. 具有列表形式的数据划分训练集和数据集：
      from sklearn.model_selection import train_test_split
      库名即方法名，注意返回的顺序是先文本后标签
   2. 统计数据中的各个类别：from collections import Counter
   3. DataLoader DataSet:
      from torch.utils.data import Dataset
      from torch.utils.data import DataLoader
   4. Linear和ReLU: import torch.nn as nn
   5. torch.optim.Adam
      torch.optim.Adam(model.parameters(),lr = lr)
   6. 只有 pathlib.Path 对象才能调用 .open() 
      from pathlib import Path
   7. f1-score:
      from sklearn.metrics import f1_score
   8. 告诉rnn不要处理padding字符产生的embedding_dim维数据：
      from torch.nn.utils.rnn import (
         pack_padded_sequence,
         pad_packed_sequence,
      )


2. 常用指令：
   1. 模型训练：model.train()
      模型评估：model.eval()
   2. 梯度清零: opti.zero_grad()
   3. 计算梯度：loss.backward()
   4. 更新参数: opti.step()
   5. 模型保存: torch.save(model.state_dict(),'../outputs/model.pt')

   6. json保存：
         vocab_path = Path('../outputs/vocab.json')
         with vocab_path.open('w',encoding='utf-8') as file:   
         json.dump(
            char_to_id,
            file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )

   7. json加载：
      vocab_json_path = Path("../outputs/vocab.json")
      with vocab_json_path.open('r',encoding='utf-8') as file:
         char_to_id = json.load(file)

   8. 模型加载：
      with best_model_conf_path.open('r',encoding='utf-8') as best_model_conf_file:
         best_model_conf = json.load(best_model_conf_file)

      best_model = TicketClassifierMLP(vocab_size=best_model_conf['vocab_size'],
                                       embedding_dim=best_model_conf['embedding_dim'],
                                       padding_id=best_model_conf['padding_id'],
                                       hidden_dim=best_model_conf['hidden_dim'],
                                       dropout=best_model_conf['dropout'],
                                       num_classes=best_model_conf['num_classes'])

      dict = torch.load(best_model_path)
      '''用模型接受参数'''
      best_model.load_state_dict(dict)

   9. 构造rnn:
         rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,
                  num_layers=3,batch_first=True,
                  nonlinearity='tanh')
   10. 将dataframe保存csv:
         class_distribution.to_csv(class_distribution_path,index=False,
                              encoding='utf-8')
   11. 保存与加载sklearn的模型
         import joblib
         model_path = outputs_path / "model.joblib"
         model = results["model"]
         joblib.dump(model, model_path)
      
         model = joblib.load(model_path)
   12. 创建TF-IDF编码器：
         from sklearn.feature_extraction.text import TfidfVectorizer
         tfidf = TfidfVectorizer(analyzer='char',ngram_range=(1,2))
   13. TF-IDF的训练集训练编码与验证集、测试集的编码：
         # 对训练集数据编码
         train_features = tfidf.fit_transform(data['train_texts'])
         # 用训练集的词典对验证集编码
         eval_features = tfidf.transform(data['eval_texts'])
   14. 对DataFrame进行排序：
         error_cases = (error_cases.sort_values(by="confidence",ascending=False)
         .reset_index(drop=True))
   15. 创建GridSearchCV
         grid_search_cv = GridSearchCV(estimator=pipeline,
                                    param_grid=params_range,# 使用宏平均F1选择最佳参数
                                    scoring="f1_macro",cv=predefined_split,n_jobs=-1,
                                    # 选择最佳参数后,使用训练集 + 验证集重新训练最终模型
                                    refit=True)
   16. 关于随机数：
         任意尺寸的标准正态torch.randn()
         任意尺寸的范围整数text_ids = torch.randint(low=1,high=vocab_size,size=(8,32))
   
   17. Hugging Face创建官方中文Tokenizer：
   from transformers import AutoTokenizer
   MODEL_NAME = "google-bert/bert-base-chinese"
   tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

3. 修改vscode的运行目录：
   1. 让终端默认在当前Python文件所在目录运行：
      可以打开 VS Code 设置，搜索：
        Execute In File Dir
      然后勾选：
        Python > Terminal: Execute In File Dir
      这样运行 Python 文件时，终端会先切换到该文件所在目录
   2. 在 VS Code 左侧文件资源管理器中：
      找到想进入的文件夹 右键 选择 “在集成终端中打开”
      终端会自动打开，并且运行目录就是这个文件夹。
      **此操作会定位至当前项目使用的python解释器，便于安装库**

4. 代码整体向后：Tab  整体向前： Shift + Tab

5. 选择从项目的根目录(LEARNINGPLAN)运行某python文件：
   & D:\Miniconda3\envs\project1\python.exe -m week3.src.train

   1. 在week3/src/train.py中需要导入同目录的data_prepare.py:
      from .data_prepare .表示相对路径导入，即同目录下开始寻找
   
   2. 在week3/src/train.py中需要导入不同目录但是不跨顶层目录week3的week3/src/models/mlp.py中的类：
      from .models.mlp .表示同目录下models目录中mlp.py
   
   3. 在week3/src/data_prepare.py中需要导入不同目录跨顶层目录week3的common/dataset.py中的类：
      from common.dataset import 使用绝对路径导入，此时python会从搜索路径出发（从根目录LEARNINGPLAN出发找common）
   
   4. 如果直接从具体文件中运行（点三角），
      1. 那么它就不知道你的 . 是什么意思，因为.表示当前包，但这个文件的当前包是什么。**ImportError: attempted relative import with no known parent package**
      2. 绝对路径导入是，此时搜索路径就变成了week3/src/，如果导入from common.dataset import ，它就会去week3/src/里面找common,自然找不到


6. 关于文件路径：使用 __file__ 可以让程序不依赖“从哪里启动”，通常更加稳定
   不管是在根目录中运行还是直接在具体文件中运行，Path(__file__)始终定位在文件的绝对路径
   

