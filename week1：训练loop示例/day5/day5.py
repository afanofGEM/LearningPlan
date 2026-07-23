import torch
import torch.nn as nn
import torch.optim as optim
import os

class MyMLP(nn.Module):
    def __init__(self, input_features, hidden_features,output_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_features)
        )

    def forward(self, x):
        return self.net(x)
    

# 数据->模型（得到预测值）-> 计算损失-> 更新参数 -> 重复几次
def train(x_train,y_train,model,loss_fn,optimizer,epochs):
    model.train() # 训练模式

    for epoch in range(epochs):

        optimizer.zero_grad() # 梯度清零

        y_pred = model(x_train)
        loss = loss_fn(y_pred,y_train)
        
        if (epoch+1) % 10 == 0:
            print(f"epoch {epoch+1}: loss = {loss.item()}")
            
        loss.backward()
        optimizer.step()

    os.makedirs("week1/output",exist_ok=True)
    torch.save(model.state_dict(),"week1/output/MLPModel.pth")
    print("Model saved to output/MLPModel.pth")


def evaluate(x_test,y_test,model,loss_fn):
    model.eval() # 测试模式

    with torch.no_grad(): # 告诉模型不需要更新了

        y_pred = model(x_test)
        loss = loss_fn(y_pred,y_test)

        y_pred_class = torch.argmax(y_pred,dim=1) # 求每一列最大值，就是每个样本的预测分类
        acc = (y_pred_class == y_test).float().mean() # 计算准确率
        print(f"Test Loss: {loss.item()}, Test Accuracy: {acc.item()}")


def main():
    x = torch.rand(300,4)
    y = torch.randint(0,3,(300,))
    x_train = x[:240]
    y_train = y[:240]
    x_test = x[240:]
    y_test = y[240:]

    model = MyMLP(input_features=4,hidden_features=16,output_features=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01) # adam会动态更新lr
    epochs = 100
    train(x_train,y_train,model,loss_fn,optimizer,epochs)
    evaluate(x_test,y_test,model,loss_fn)


if __name__ == "__main__":
    main()




