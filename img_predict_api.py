import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


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

# CSV 파일을 통해 라벨 맵 생성
csv_file = r'C:\Users\PC\OneDrive\Desktop\Project\argumented_data.csv'
df = pd.read_csv(csv_file)
label_map = {i: label for i, label in enumerate(df['label'].astype('category').cat.categories)}

# 저장된 모델 불러오기
num_classes = len(label_map)
model = CNN(num_classes=num_classes)
model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
model.eval()

# 이미지 예측 함수
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_transformed = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    with torch.no_grad():
        outputs = model(image_transformed)
        _, predicted = torch.max(outputs, 1)
        
    return label_map[predicted.item()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    # POST 요청에서 이미지 파일 받기
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400
    image_file = request.files['image']

    try:
        filename = secure_filename(image_file.filename)
        file_path = os.path.join('uploads', filename)

        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        image_file.save(file_path)

        label = predict_image(image_file)
        return jsonify({'label' : label})
    except Exception as e:
        app.logger.error(f"Exception occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug= True)
   


# API로 json 파일 내보내기
#https://devrokket.tistory.com/3 (참고)
#https://tutorials.pytorch.kr/intermediate/flask_rest_api_tutorial.html



# 1.23.5