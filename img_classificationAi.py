import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as ts
import torch
import torch.nn as nn
import torch.optim as optim
# 랜덤 시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class CustomDataSet(Dataset):
    def __init__(self,dataframe,image_dir,transform =None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,self.dataframe.iloc[idx,0])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx,1]

        if self.transform:
            image = self.transform(image)

        return image, label
    
# 데이터 전처리

transform = ts.Compose([
    ts.Resize((256,256)),
    ts.ToTensor(),
    ts.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

image_dir = r'C:\Users\PC\OneDrive\Desktop\Project\crop_imgs(1)'
csv_file = r'C:\Users\PC\OneDrive\Desktop\Project\argumented_data.csv'


df = pd.read_csv(csv_file)
df['label'] = df['label'].astype('category').cat.codes
train_df, test_df = train_test_split(df, test_size=0.2,stratify=df['label'],random_state=42)

class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1. / class_counts
sample_weights = class_weights[train_df['label']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset = CustomDataSet(train_df,image_dir,transform=transform)
test_dataset = CustomDataSet(test_df, image_dir, transform=transform)

train_loader = DataLoader(train_dataset,batch_size=32,sampler=sampler,num_workers=4)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=4)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 32 * 32, 1000)
        self.fc2 = nn.Linear(1000,num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out= self.dropout(out)
        out = self.fc2(out)

        return out
    
if __name__ == '__main__':
        
    num_classes = len(df['label'].unique())
    model = CNN(num_classes=num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    num_epochs = 6
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images,labels = images.to(device), labels.to(device,dtype = torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)


         # 검증 데이터셋에서 손실 계산
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.long)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    print("Training finished.")

    # 모델 평가
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device,dtype = torch.long)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'F1 Score: {f1:.3f}')


    torch.save(model.state_dict(), 'trained_model.pth')
    print("모델 저장")