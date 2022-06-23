# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:47:48 2022

這個檔案主要是拿來訓練圖片輸出然後標籤
步驟:
    1. 準備一個如下的資料結構
        train_img (Folder)
            ------ label_n_name (Folder)
                ------ image_n.jpg / image_n.png / image_n.bmp (file)
    2. 使用這裡的 readImgFolder()   可以放在 步驟 3 內
    3. from torch.utils.data import Dataset  ==> 自己寫一個處理 data的 class 
    4. torch.utils.data.DataLoader
    5. 使用這裡的 train()
                

@author: Yuyi
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def readImgFolder(path):
    """ 
    Parameters
    -------------
    path: 存放圖片的資料夾(資料夾名稱是圖片標籤名)的父資料夾
    
        資料夾 (path)
        ------資料夾 n (會被當作標籤名稱)
               ------image.jpg
               ------image.bmp
               ------image.png
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"readImgFolder(path) 找不到此資料夾!! ==> {path} ")
    path = os.path.abspath(path)
    label_list = os.listdir(path)
    label_list.sort()
    output_list = []
    for label in label_list:
        imgs_path = os.path.join(path, label)   ## 圖片所在資料夾的路徑
        if os.path.isdir(imgs_path):
            img_list = os.listdir(imgs_path)
            for img in img_list:
                full_path = os.path.join(imgs_path, img)  ## 包含圖片名稱的完整路徑
                if os.path.isfile(full_path):
                    ext = img.split(".")[-1].lower()
                    if ext=="jpg" or ext=="png" or ext=="bmp":
                        output_list.append([full_path, label])
    # print(output_list)
    return output_list



class MyCrossEntropy(nn.Module):
    def __init__(self):
        super(MyCrossEntropy, self).__init__()
    
    def forward(self, pred, y):
        batch_loss = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pred = pred.to(device)
        for i in range(pred.shape[0]):
            numerator = torch.exp(pred[i, y[i]])
            denominator = torch.sum(torch.exp(pred[i, :]))
            loss = -torch.log(numerator / denominator)
            batch_loss += loss
            
        return batch_loss / (i+1)
    
def train_one_epoch(train_loader, model, loss_fn, optimizer, device):
    size = len(train_loader.dataset)
    loss_v = 0
    acc_v = 0
    # model.requires_grad_(True)
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        # pred = pred.type(torch.long)
        # y = y.type(torch.long)
        loss = loss_fn(pred, y)
        loss_v += loss.item()
        acc_v += (pred.argmax(1)==y).float().sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            current = (batch+1) * len(X)
            print(f"Batch: {batch+1} ==> loss: {loss.item():>7f} [{current:> 5d} / {size:>5d}]")
    
    loss_v /= (batch+1)
    acc_v /= size
    return model, loss_v, acc_v 

def test(test_loader, model, loss_fn, device):
    size = len(test_loader.dataset)
    loss_v = 0
    acc_v = 0
    model.eval()
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss_v += loss.item()
            acc_v += (pred.argmax(1)==y).float().sum().item()
            
    loss_v /= len(test_loader)
    acc_v /= size
    return loss_v, acc_v 

def plot_on_air(train_loss_list, train_acc_list, test_loss_list, test_acc_list, total_epochs=None, save_path="./"):
    x = [i for i in range(1, len(train_loss_list)+1)]
    fig = plt.figure("loss-acc", figsize=(10, 5))
    fig.clf()  ## clear figure
    fig.gca()
    ax = plt.subplot(1, 2, 1)
    plt.plot(x, train_loss_list, c='r', label="train_loss")
    plt.plot(x, test_loss_list, c='b', label="test_loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.tight_layout()
    plt.legend()
    if total_epochs != None:
        plt.xlim((0, total_epochs+1))
    
    ax = plt.subplot(1, 2, 2)
    plt.plot(x, train_acc_list, c='r', label="train_acc")
    plt.plot(x, test_acc_list, c='b', label="test_acc")
    plt.title("Acc")
    plt.xlabel("epochs")
    plt.tight_layout()
    plt.legend()
    
    plt.ylim((0, 1))
    if total_epochs != None:
        plt.xlim((0, total_epochs+1))
    
    plt.savefig(os.path.join(save_path, "loss_acc.png"))
    plt.show()
    

def train(train_loader, test_loader, model, epochs=10, lr=0.01, opt="sgd", early_stop=200, model_path="./model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("Using:",device)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = MyCrossEntropy()
    if opt.lower()=="sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    ## record
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = list(), list(), list(), list()
    best_acc = 0
    no_improve = 0
    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}----------------------------------------------")
        model, train_loss, train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test(test_loader, model, loss_fn, device)
        print(f"\tResult: loss: [{train_loss}, {test_loss}],  acc: [{train_acc*100:>.2f}%, {test_acc*100:>.2f}%]")
        
        ## 降低學習率
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr > 1e-5:
            stepLR.step()
        print(f"\t\t學習率{current_lr}")
        
        ## record
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        ## plot on air
        plot_on_air(train_loss_list, train_acc_list, test_loss_list, test_acc_list, total_epochs=epochs, save_path=model_path)
        
        ## save
        ##### save best
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path+"best.pth")
            print(f"save best model!! acc: {best_acc*100:>.2f}%")
            no_improve = 0
        else:
            no_improve += 1
        
        ##### save early stop
        if no_improve > early_stop:
            torch.save(model.state_dict(), model_path+"final_early_stop_"+str(round(test_acc*100, 2))+".pth")
            print("Save final early stop model!!",f"acc: {test_acc*100:>.2f}%")
            return
    
    torch.save(model.state_dict(), model_path+"final_"+str(round(test_acc*100, 2))+".pth")
    print(f"save final model!! acc: {test_acc*100:>.2f}%")
    return model


def main():
    pass


if __name__ == "__main__":
    main()