import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 파일 경로 설정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


# 모델 정의 (훈련할 때 사용한 것과 동일해야 함)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 32 * 32, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 데이터 전처리 정의 (훈련할 때 사용한 것과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 예측 함수
def predict_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_transformed = transform(image).unsqueeze(0)  # 배치 차원 추가
    image_transformed = image_transformed.to(device)
    
    with torch.no_grad():
        outputs = model(image_transformed)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item(), image

def plot_image_and_prediction(image, predicted_label, label_map):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f'Predicted Label: {label_map[predicted_label]}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # CSV 파일을 통해 라벨 맵 생성
    csv_file = r'C:\Users\PC\OneDrive\Desktop\Project\argumented_data.csv'
    df = pd.read_csv(csv_file)
    label_map = {i: label for i, label in enumerate(df['label'].astype('category').cat.categories)}

    # 저장된 모델 불러오기
    num_classes = len(label_map)
    model = CNN(num_classes=num_classes)
    model.load_state_dict(torch.load('trained_model.pth'))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 테스트할 이미지 경로
    test_image_path = r'C:\Users\PC\OneDrive\Desktop\Project\화장대.jpg'
    
    # 예측 및 시각화
    predicted_label, image = predict_image(model, test_image_path, transform, device)
    
    plot_image_and_prediction(image, predicted_label, label_map)
