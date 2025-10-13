import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

# Step1: 自定义定义CarDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class CarDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        #self.images_file = images_file
        self.transform = transform
        self.image_list = []
        self.label_list = []
        # 设置图片
        for root, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                self.image_list.append(fname)
        # 设置label
        data = pd.read_csv(labels_file, header=None)
        self.label_list = data[0].tolist()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.root_dir + self.image_list[idx]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = Image.open(image_path)
        
        # 如果是灰度图像，转换为RGB
        if image.getbands()[0] == 'L':
             image = image.convert('RGB')        
                
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}

#train_data = CarDataset('/nas/stanford_car_dataset/cars_train', labels_file='/nas/stanford_car_dataset/devkit/train_perfect_preds.txt')
#print(train_data.image_list)

import torch.nn as nn
import torch.optim as optim

# 超参数定义
EPOCH = 5              # 训练epoch次数
BATCH_SIZE = 64         # 批训练的数量
LR = 0.001              # 学习率
DOWNLOAD_MNIST = False  # 设置True 可以自动下载数据

my_tf = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor()])

train_data = CarDataset('/nas/stanford_car_dataset/cars_train/', labels_file='/nas/stanford_car_dataset/devkit/train_perfect_preds.txt', 
                        transform = my_tf)

train_loader = DataLoader(dataset=train_data, batch_size = 16, shuffle=True)
#model = torchvision.models.resnet50(pretrained=False)
model = torchvision.models.densenet169(pretrained=False)

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 训练
for epoch in range(EPOCH):
    for i, data in enumerate(train_loader):
        inputs, labels = data['image'], data['label']
        #print(inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch{} loss:{:.4f}'.format(epoch+1, loss.item()))
    
file_name = 'stanford_car_densenet.pt'
torch.save(model, file_name)

# 模型加载
model = torch.load(file_name)
# 测试
model.eval()
image = Image.open('./149-1.jpg')
# 如果是灰度图像，转换为RGB
if image.getbands()[0] == 'L':
     image = image.convert('RGB')        
image = my_tf(image)
print(image.shape)
image = image.unsqueeze(0)
image = image.to(device)
print(image.shape)
# 前向传播
outputs = model(image)
_, predicted = torch.max(outputs.data, 1)
print(predicted.item())
