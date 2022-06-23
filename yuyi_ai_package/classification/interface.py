# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:44:07 2022

@author: Yuyi

"""




from .classiifier import *
from torch.utils.data import Dataset
import torchvision.transforms as trans
import os
from PIL import Image 
import sklearn.metrics as metrics
import seaborn as sns
import pandas as pd

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
    
    
    def train(self, train_loader, test_loader, model, epochs=10, lr=0.0001, opt="sgd", early_stop=200):
        """ 訓練模型 """
        self.__model = train(train_loader, test_loader, model, epochs, lr, opt, early_stop, model_path=self.__folder_path)
        return self.__model 
    
    def evaluate(self, test_loader, model_name="best.pth"):
        """ 評估模型 """
        device = "cpu"
        model_path = os.path.join(self.__folder_path, model_name)
        parameters = torch.load(model_path)
        self.__model.load_state_dict(parameters)
        self.__model.to(device)
        self.__model.eval()
        total_y, total_pred = list(), list()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = self.__model(x)
                
                for i in range(len(y)):
                    total_y.append(y[i])
                    total_pred.append(pred[:].argmax(1)[i])
                
        # pred = pred[:].argmax(1)
        out1 = metrics.classification_report(total_y, total_pred, output_dict=True)
        print("Accuracy:  ",out1["accuracy"])
        # print(out1)
        
        ## 整理 precision、recall、f1-score
        self.cleanResult(self.__label_dict_reverse, out1, save_path=self.__folder_path)
        
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
        
        
    
    





