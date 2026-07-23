from .data_prepare import data_prepare
import torch
from pathlib import Path

# 解决文件路径问题
current_file_dir = Path(__file__).resolve().parent.parent
outputs_dir = current_file_dir/"outputs"
outputs_dir.mkdir(parents=True,exist_ok=True)

train_size = 0.6
seed = 15
max_len = 32
batch_size = 8
file_path = Path(__file__).parent.parent / 'data' / '工单数据集.csv'

'''{
        "train_dataloader":train_dataloader,
        "valid_dataloader":valid_dataloader,
        "test_dataloader":test_dataloader,
        "char_to_id":char_to_id,
        "label_to_id":label_to_id
    }'''
data = data_prepare(file_path,train_size=train_size,seed=seed,
                    max_len=max_len,batch_size=batch_size)


'''接收：数据，模型，损失函数，优化器
   得到：训练集每一轮的loss,acc,f1-score'''
def train_epoch(dataloader,model,loss_fn,opti):

    model.train()
    train_loss = 0.0
    train_num = 0
    corr_num = 0
    all_pred = []
    all_label = []
    for batch in dataloader:
        '''清除梯度'''
        opti.zero_grad()

        x = batch['text_idx'] # (batch_size,max_len)
        y = batch['label_idx'] # (batch_size)
        y_pred = model(x) # (batch_size,num_classes)
        class_pred = y_pred.argmax(dim=1) # 求每行的最大值，(batch_size)
        
        '''1.正确数'''
        corr_num += (class_pred==y).sum().item()

        loss = loss_fn(y_pred,y)
        num = y.size(0)
        '''2.总数'''
        train_num += num
        '''3.每轮损失'''
        train_loss += loss.item() * num

        '''4.计算梯度，更新参数'''
        loss.backward()
        opti.step()

        '''5.记录预测与真实标签'''
        all_pred.extend(class_pred.tolist())
        all_label.extend(y.tolist())
    
    acc = corr_num / train_num
    avg_loss = train_loss / train_num

    from sklearn.metrics import f1_score
    '''zero_division=0：如果某类完全没被预测到,就把它的F1记为0,然后程序不中断,继续算宏平均'''
    f1 = f1_score(all_label,all_pred,average='macro',zero_division=0)
    
    return {
        'acc':acc,
        'avg_loss':avg_loss,
        'f1':f1
    }


'''接收：数据，模型，损失函数，不需要优化器
   得到：验证集每一轮的loss,acc,f1-score'''
def eval_epoch(dataloader,model,loss_fn):
    
    model.eval()
    eval_loss = 0.0
    eval_num = 0
    corr_num = 0
    all_pred = []
    all_label = []

    with torch.no_grad():
        for batch in dataloader:

            x = batch['text_idx'] # (batch_size,max_len)
            y = batch['label_idx'] # (batch_size)
            y_pred = model(x) # (batch_size,num_classes)
            class_pred = y_pred.argmax(dim=1) # 求每行的最大值，(batch_size)
            
            '''1.正确数'''
            corr_num += (class_pred==y).sum().item()

            loss = loss_fn(y_pred,y)
            num = y.size(0)
            
            '''2.总数'''
            eval_num += num
            '''3.每轮损失'''
            eval_loss += loss.item() * num

            '''4.记录预测与真实标签'''
            all_pred.extend(class_pred.tolist())
            all_label.extend(y.tolist())
        
        acc = corr_num / eval_num
        avg_loss = eval_loss / eval_num

        from sklearn.metrics import f1_score
        '''zero_division=0：如果某类完全没被预测到,就把它的F1记为0,然后程序不中断,继续算宏平均'''
        f1 = f1_score(all_label,all_pred,average='macro',zero_division=0)
        
    return {
        'acc':acc,
        'avg_loss':avg_loss,
        'f1':f1
    }


'''
实验	Learning rate	Dropout
A	1e-3	0.0
B	1e-3	0.3
C	1e-2	0.3'''
from .models.mlp import classifier_mlp
import torch.nn as nn


'''规定了超参数的一次实验的训练过程：
   进行多轮的训练集训练与验证集验证
   一轮训练集的作用：更新模型的参数，得到当前的loss,acc,f1
   一轮验证集的作用：根据验证集的f1来保存最佳模型，保存当前的loss,acc,f1
   上面的6个变量全部被储存至history的6个list中，list的每个元素对应一轮的数据
   得到:过程量history，结果量：最优的验证集loss,acc,f1，保存模型'''
def run_experiment(data,lr,dropout,num_epoch,experiment_name):

    model = classifier_mlp(len(data['char_to_id']),embedding_dim=64,
                           padding_id=data['char_to_id']['<PAD>'],
                           hidden_dim=128,num_classes=len(data['label_to_id']),
                           dropout=dropout)
    
    loss_fn = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(model.parameters(),lr = lr)
    save_path = outputs_dir / f"best_model_{experiment_name}.pt"


    '''{
        'best_epoch':best_epoch,
        'best_eval_loss':best_eval_loss,
        'best_eval_acc':best_eval_acc,
        'best_eval_f1':best_eval_f1,
        'history':history
    }'''
    print(f"{experiment_name}超参数组合下的模型训练开始：")

    history = {
        'train_loss':[],
        'train_acc':[],
        'train_f1':[],
        'eval_loss':[],
        'eval_acc':[],
        'eval_f1':[]
    }
    best_epoch = 0
    best_eval_loss = float('inf')
    best_eval_acc = 0.0
    best_eval_f1 = -1.0
    train_dataloader = data['train_dataloader']
    eval_dataloader = data['valid_dataloader']

    for epoch in range(num_epoch):
        train_dict = train_epoch(dataloader=train_dataloader,model=model,
                                 loss_fn=loss_fn,opti=opti)
        history['train_loss'].append(train_dict['avg_loss'])
        history['train_acc'].append(train_dict['acc'])
        history['train_f1'].append(train_dict['f1'])

        eval_dict = eval_epoch(dataloader=eval_dataloader,model=model,
                            loss_fn=loss_fn)
        history['eval_loss'].append(eval_dict['avg_loss'])
        history['eval_acc'].append(eval_dict['acc'])
        history['eval_f1'].append(eval_dict['f1'])

        print(
            f"Epoch {epoch + 1:02d}/{num_epoch} | "
            f"train loss: {train_dict['avg_loss']:.4f} | "
            f"valid loss: {eval_dict['avg_loss']:.4f} | "
            f"train acc: {train_dict['acc']:.4f} | "
            f"valid acc: {eval_dict['acc']:.4f} | "
            f"train F1:{train_dict['f1']:.4f} |"
            f"valid F1: {eval_dict['f1']:.4f}"
        )
        
        '''根据验证集f1保存最佳模型'''
        if eval_dict['f1'] > best_eval_f1:
            best_epoch = epoch + 1
            best_eval_loss = eval_dict['avg_loss']
            best_eval_acc = eval_dict['acc']
            best_eval_f1 = eval_dict['f1']
        
            torch.save(model.state_dict(),save_path)
    
    return {
        'best_epoch':best_epoch,
        'best_eval_loss':best_eval_loss,
        'best_eval_acc':best_eval_acc,
        'best_eval_f1':best_eval_f1,
        'history':history,
        'lr':lr,
        'dropout':dropout,
        'experiment_name':experiment_name,
        'save_path':save_path
    }


'''不同的超参数组合'''
def main():

    experiments = [
        {
            "experiment_name": "A",
            "lr": 1e-3,
            "dropout": 0.0,
        },
        {
            "experiment_name": "B",
            "lr": 1e-3,
            "dropout": 0.3,
        },
        {
            "experiment_name": "C",
            "lr": 1e-2,
            "dropout": 0.3,
        },
    ]

    results = []

    for conf in experiments:
        '''result={
            'best_epoch':best_epoch,
            'best_eval_loss':best_eval_loss,
            'best_eval_acc':best_eval_acc,
            'best_eval_f1':best_eval_f1,
            'history':history
            'lr':lr,
            'dropout':dropout,
            'experiment_name':experiment_name,
            'save_path':save_path
        }'''
        result = run_experiment(data=data,lr=conf['lr'],dropout=conf['dropout'],
                       num_epoch=20,experiment_name=conf['experiment_name'])
        results.append(result)

    best_result = max(
        results,
        key= lambda x : x['best_eval_f1']
    )

    # 得到与result相同格式的最佳结果
    save_path = outputs_dir / 'best_model.pt'

    import shutil
    shutil.copyfile(best_result['save_path'],save_path)

    # 输出最佳的实验名称，实验超参数
    print("\n最终选择：") 
    print(f"实验 {best_result['experiment_name']}" ) 
    print(f"learning rate：" f"{best_result['lr']}" ) 
    print(f"dropout：{best_result['dropout']}" ) 
    print(f"验证集 F1：" f"{best_result['best_eval_f1']:.4f}" ) 
    print( f"最终模型已保存到：{save_path}")

    # 保存最佳模型的history
    import json
    history_path = Path(__file__).parent.parent / "outputs" / "history.json"
    with history_path.open('w',encoding='utf-8') as file:   
        json.dump(
            best_result['history'],
            file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )

    # 保存最佳模型的其余配置，这个是便于后续加载模型时进行模型的初始化
    best_model_conf = {
        'train_size' : train_size ,
        'seed': seed,
        'max_len':max_len,
        'batch_size': batch_size,
        'embedding_dim':64,
        'hidden_dim':128,
        'lr':best_result['lr'],
        'dropout':best_result['dropout'],
        'experiment_name':best_result['experiment_name'],
    }

    best_model_conf_path = Path(__file__).parent.parent / "outputs" / "best_model_config.json"
    with best_model_conf_path.open('w',encoding='utf-8') as file:   
        json.dump(
            best_model_conf,
            file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )

if __name__ == '__main__':
    main()

