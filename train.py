from data import *
from torch.utils.data import DataLoader
import torch
from model.FM import FM
from model.MLP import MLP
from model.DeepFM import DeepFM
import torch.nn as nn
import torch.optim as optim
from run import *
from utils.load_model_param import load_model_param
import argparse


def main():

    #模型字典，模型名作为参数输入，根据value值返回对应模型
    model_dict = {
        "FM": FM,
        "MLP": MLP,
        "DeepFM": DeepFM
    }

    #参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='Movielens1M',help='Movielens1M')
    parser.add_argument('--wd', type=float, default=1e-3, help='the weight decay of optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
    parser.add_argument('--gpu_index', type=int, default=0, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument("--seed", default='1')
    parser.add_argument('--model_name',type=str,default='FM',help='MLP/FM/DeepFM')
    params = parser.parse_args()

    #设置随机种子，确保模型可复现
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    model_name = params.model_name
    
    #加载json格式数据
    model_param = load_model_param("config/"+model_name+".json")

    if params.dataset_name == 'Movielens1M' :
        #MovieLens1M数据集
        #userID itemID label
        dataset = MovieLensDataset('dataset/MovieLens1M/ratings.dat')
        num_features= dataset.field_dims
        train_length = int(len(dataset) * 0.8)
        test_length = len(dataset)-train_length
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, test_length))
    
    #数据集、模型及超参数信息写入日志文件
    f = open("./log/"+model_name+'/'+params.dataset_name+'/' + str(params.batch_size) + '_' +  str(params.wd) + '_' +
             str(params.lr) +'_'+ str(params.seed) +'_' + "log.txt", "w")
    f.write(str(params)+'\n')
    f.write(str(model_param)+'\n')
    
    #获得批次训练数据
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size)
    #print(len(train_loader))
    device = torch.device("cuda:{}".format(params.gpu_index) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    #获得指定的模型
    model = model_dict[model_name](num_features,model_param)
    
    #指定损失函数
    criterion = nn.BCELoss()
    
    #指定优化器
    optimizer = optim.Adam(model.parameters(), lr=params.lr,weight_decay=params.wd)
    model.to(device)
    #训练
    num_epochs = 10
    for epoch in range(num_epochs):
        #训练集进行训练
        train_loss = train(model, train_loader, criterion, optimizer,device)
        #测试集返回AUC性能指标
        auc = evaluate(model, test_loader,device)
        #输出到终端以及日志文件中
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, AUC: {auc:.4f}")
        f.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, AUC: {auc:.4f}\n")
    f.close()
if __name__ == '__main__':
    main()