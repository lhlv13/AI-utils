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