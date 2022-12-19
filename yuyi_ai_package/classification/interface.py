# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:44:07 2022

@author: Yuyi

"""

""" 使用方式
import yuyi_ai_package as yuyi
import torch, torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

#################################################################################################
## 修改成自己的路徑
train_img_path = r".\hymenoptera_data\train"
val_img_path = r".\hymenoptera_data\val"
test_img_path = r".\hymenoptera_data\val"
channel = 3
#################################################################################################

def main():
    ## Transforms
    data_transforms = {
            "train" :transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
            "val" : transforms.Compose([
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        }
    
    
    ## DataLoader
    train_dataset = yuyi.classification.Classify_Dataset(train_img_path, transforms=data_transforms["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=32,
                                                shuffle=True,
                                                drop_last=False)
    print("train_loader successful!!")
    
    val_dataset = yuyi.classification.Classify_Dataset(val_img_path, transforms=data_transforms["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=32,
                                                shuffle=True,
                                                drop_last=False)
    print("val_loader successful!!")
    
    test_dataset = yuyi.classification.Classify_Dataset(test_img_path, transforms=data_transforms["val"])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=len(test_dataset),
                                                shuffle=True,
                                                drop_last=False)
    print("test_loader successful!!")
    
    ## Label  {標籤 : 數字}
    label_dict = train_dataset.getLabeldict()
    total_class_num = len(label_dict)
    ## 模型
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    ## model.conv1 = nn.Conv2d(channel, 64, 7, stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, total_class_num)
    
    ## 訓練
    epochs = 10
    lr = 0.001
    opt = "adam"
    classifyObj = yuyi.classification.Classify(label_dict)
    classifyObj.train(train_loader, val_loader, model, epochs=epochs, lr=lr, opt=opt)
    
    ## 評估
    classifyObj.evaluate(test_loader)
    
    
if __name__ == "__main__":
    main()
"""


from .classiifier import *
from torch.utils.data import Dataset
import torchvision.transforms as trans
import os
from PIL import Image 
import sklearn.metrics as metrics
import seaborn as sns
import pandas as pd
import json

class Classify_Dataset(Dataset):
    """ 
    torch 的 dataset處理
    """
    def __init__(self, img_folder, input_img_size=(64, 64), transforms=None):
        """
        Input:
        ----------
        img_folder: 路徑資料夾內放有許多以標籤名稱命名的資料夾，資料夾放每一類的 images
        input_img_size: 輸入圖片的大小，需與自己建構的模型一樣
        transforms: 自定義的 transforms
        """
        if not os.path.isdir(img_folder):
            raise FileNotFoundError("path is error!!!")
        
        self.__label_name = {}
        self.__img_list = readImgFolder(img_folder)
        self.__input_img_size = input_img_size
        self.__transforms = transforms 
        self.len = len(self.__img_list)
        
        ## 將 label 轉成 數字
        label_list = os.listdir(img_folder)
        label_list.sort()
        label_num = 0
        for label in label_list:
            if(os.path.isdir(os.path.join(img_folder, label))):
                self.__label_name[label] = label_num
                label_num += 1
                
            
            
        
        
    def __getitem__(self, index):
        img_path, label = self.__img_list[index][0], self.__img_list[index][1]
        img = Image.open(img_path)
        if self.__transforms:
            img = self.__transforms(img)
        else:
            resize = trans.Compose([
                            trans.Resize([self.__input_img_size[0], self.__input_img_size[1]]),
                            trans.ToTensor()])
            img = resize(img)
        # img = torch.unsqueeze(img, dim=0)  ## 有 totensor 就不用這個了
        label = self.__label_name[label]
        return img, label
    
    def __len__(self):
        return self.len
    
    def getLabeldict(self):
        return self.__label_name


#####################################################################
class Classify():
    """ 包含訓練，評估模型 ， 使用前要先 dataloader """
    def __init__(self, label_dict):
        """  
        Input:
        ----------
        label_dict: 可從 Classify_Dataset 類別 取得， 結構為 ==> {標籤名 : 對應數字}
        """
        self.__model = None
        self.__label_dict = label_dict  ## {標籤n : 數字n}
        self.__label_dict_reverse = {}
        for key, value in self.__label_dict.items():
            self.__label_dict_reverse[value] = key
        ## 儲存
        metrics_folder = "./metrics"
        if not os.path.isdir(metrics_folder):
            os.mkdir(metrics_folder)
        
        folder_name = len(os.listdir(metrics_folder))
        self.__folder_path = os.path.join(metrics_folder, f"Result_{folder_name}/")
        os.mkdir(self.__folder_path)
        
        ## 儲存 label_dict
        with open(os.path.join( self.__folder_path, "label_dict.json"), 'w') as f:
            json.dump([self.__label_dict, self.__label_dict_reverse], f)
    
    
    def train(self, train_loader, test_loader, model, epochs=10, lr=0.0001, opt="sgd", early_stop=200, input_num=1, step_size=5):
        """ 訓練模型 """
        self.__input_num = input_num
        self.__model = train(train_loader, test_loader, model, epochs, lr, opt, early_stop, model_path=self.__folder_path, input_num=self.__input_num, step_size=step_size)
        return self.__model 
    
    def evaluate(self, test_loader, model_name="best.pth", device="cpu", input_num=1):
        """ 評估模型 """
        device = device
        dtype = torch.cuda.FloatTensor if device=="cuda" else torch.FloatTensor 
        model_path = os.path.join(self.__folder_path, model_name)
        parameters = torch.load(model_path)
        self.__model.load_state_dict(parameters)
        self.__model.to(device)
        self.__model.eval()
        total_y, total_pred = list(), list()
        with torch.no_grad():
            for x, y in test_loader:
                if input_num > 1:
                    for i in range(self.__input_num):
                        x[i] = x[i].to(device).type(dtype)
                else:
                    x = x.to(device).type(dtype)
                y = y.to(device)
                pred = self.__model(x)
                
                for i in range(len(y)):
                    total_y.append(y[i])
                    total_pred.append(pred[:].argmax(1)[i].detach())
        # pred = pred[:].argmax(1)
        out1 = metrics.classification_report(total_y, total_pred, output_dict=True)
        print("Accuracy:  ",out1["accuracy"])
        # print(out1)
        
        ## 整理 precision、recall、f1-score
        metrics_dict = self.cleanResult(self.__label_dict_reverse, out1, save_path=self.__folder_path)
        
        ## 畫圖
        label_names = [key for key in self.__label_dict]
            
        
        sns.set()
        y_true = []
        for y_unit in total_y:
            y_true.append(self.__label_dict_reverse[int(y_unit)])
        y_pred = []
        for y_unit in total_pred:
            y_pred.append(self.__label_dict_reverse[int(y_unit)])
        cm = metrics.confusion_matrix(y_true, y_pred, labels=label_names)
        # cm = metrics.confusion_matrix(y_true, y_pred)
        # print(cm)
        sns.heatmap(cm, annot=True, cmap="Pastel1", fmt=".20g", xticklabels=label_names, yticklabels=label_names)
        
        ## 儲存
        
        
        plt.savefig(os.path.join(self.__folder_path,"confusion.png"))
        return metrics_dict
        
        
    def cleanResult(self, label_dict_reverse, metrics, save_path):
        metrics_dict = {"label":[], "precision":[], "recall":[], "f1-score":[]}
        for key, value in metrics.items():
            if key.isdigit():
                metrics_dict["label"].append(label_dict_reverse[int(key)])
                metrics_dict["precision"].append(value["precision"])
                metrics_dict["recall"].append(value["recall"])
                metrics_dict["f1-score"].append(value["f1-score"])
        
        ## show
        data = pd.DataFrame(metrics_dict)
        print(data)
        data.to_csv(os.path.join(save_path, "metrics.csv"))
        return metrics_dict  
    
    
    def saveError(self, test_loader, model_name="best.pth", device="cpu", input_num=1):
        """ 評估模型 """
        device = device
        dtype = torch.cuda.FloatTensor if device=="cuda" else torch.FloatTensor 
        model_path = os.path.join(self.__folder_path, model_name)
        parameters = torch.load(model_path)
        self.__model.load_state_dict(parameters)
        self.__model.to(device)
        self.__model.eval()
        path = os.path.join(self.__folder_path, "errorImage")
        count = 0
        if not os.path.isdir(path):
            os.mkdir(path)
        with torch.no_grad():
            for x, y in test_loader:
                if input_num > 1:
                    for i in range(self.__input_num):
                        x[i] = x[i].to(device).type(dtype)
                else:
                    x = x.to(device).type(dtype)
                y = y.to(device)
                pred = self.__model(x)
                
                ## 儲存錯誤分類的圖片
                for i in range(len(y)):
                    prediction = pred[:].argmax(1)[i].detach()
                    real = y.detach().numpy()[i]    
                    if prediction != real:
                        # raise Exception(x.shape)
                        img_path = os.path.join(path, f"real_{self.__label_dict_reverse[int(real)]}")
                        if not os.path.isdir(img_path):
                            os.mkdir(img_path)
                        plt.clf()
                        plt.plot(x.detach()[i][0])
                        plt.title(f"{self.__label_dict_reverse[int(real)]}>>{self.__label_dict_reverse[int(prediction)]}")
                        plt.savefig(os.path.join(img_path, f"{count}.jpg"))
                        count += 1
                    
                    
                
        print("saveError Success!!")
        
    
    





