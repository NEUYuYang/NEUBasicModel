import torch
from sklearn.metrics import roc_auc_score

#利用训练集进行训练
def train(model, train_loader, criterion, optimizer,device):
    model.train()
    running_loss = 0.0
    for x, label in train_loader:
        x = x.to(device)
        label = label.to(device)
        #print(user_id.shape)
        #print(item_id.shape)
        #print(history.shape)
        #print(item_cate.shape)
        #print(history_cate.shape)
        #concatenated_tensor = torch.cat([user_id, item_id,history,item_cate,history_cate], dim=0)
        #print(concatenated_tensor)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #print(loss.item())
    return running_loss / len(train_loader)

#利用测试集进行评估
def evaluate(model, test_loader,device):
    model.eval()
    predictions = []  
    targets = []
    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            label = label.to(device)
            output = model(x)
            predictions.extend(output.tolist())
            targets.extend(label.tolist())
    auc = roc_auc_score(targets, predictions)
    return auc