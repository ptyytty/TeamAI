import os
import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Transform 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weight_file_path = os.path.join(r"D:\work", "best_resnet.pth")
csv_file = r"D:\work\csv\label.csv"
df2 = pd.read_csv('Trash.csv')  # Assuming Trash.csv is present

# CSV 파일을 통해 라벨 맵 생성
df = pd.read_csv(csv_file)
label_map = {row['category']: row['category_name'] for _, row in df.iterrows()}

num_classes = len(label_map)

# 모델 초기화 및 가중치 로드
model = models.resnet50(pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(weight_file_path, map_location=torch.device('cpu')))
model.eval()

# 이미지 예측 함수
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_transformed = transform(image).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        outputs = model(image_transformed)
        _, predicted = torch.max(outputs, 1)

    return label_map[predicted.item()]

@app.route('/')
def upload_page():
    return render_template('index1.html')

@app.route('/fileUpload', methods=['POST'])
def upload_file():
    f = request.files.get('file')  # 'file' 필드에서 파일 가져오기
    if f and f.filename:  # 파일이 제대로 업로드 되었는지 확인
        if f:
            print(f"Uploaded file: {f.filename}")  # 파일 이름 출력
        else:
            print("No file uploaded.")  # 파일이 없을 경우 메시지 출력
        filename = secure_filename(f.filename)
        file_path = os.path.join(r'C:\Users\PC\OneDrive\Desktop\Project\static\uploads', filename)
        if not os.path.exists(r'C:\Users\PC\OneDrive\Desktop\Project\static\uploads'):
            os.makedirs(r'C:\Users\PC\OneDrive\Desktop\Project\static\uploads')
        f.save(file_path)

        try:
            label = predict_image(file_path)
            print(f"Predicted label: {label}")

            filtered_data = df2[df2['품명'] == label]
            print(f"Filtered data: {filtered_data}")

            if filtered_data.empty:
                print("Filtered data is empty. No matching label found.")
                return "No matching data found.", 404  # 적절한 응답 반환

            filtered_data.to_csv('ttrash.csv', index=None)
            print("ttrash.csv 파일이 생성되었습니다.")

            data = pd.read_csv('ttrash.csv')
            return render_template('view.html', tables=[data.to_html()], titles=[''], label=label, image_filename=filename)

        except Exception as e:
            print(f"Error occurred during file processing: {e}")
            return "An error occurred during file processing.", 500  # 적절한 응답 반환

    return "No file uploaded.", 400  # 파일이 업로드되지 않았을 때 응답


if __name__ == '__main__':
    app.run(debug=True)
