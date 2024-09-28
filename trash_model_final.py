# 이 스크립트는 이미지 분류를 위한 다양한 딥러닝 모델을 학습하고 평가하는 작업을 수행함.
# PyTorch를 기반으로 EfficientNet, ViT(비전 트랜스포머), ResNet, SENet, ResNet-SENet 등 여러 최신 신경망 모델을 활용하며,
# 데이터 로드, 전처리, 모델 학습, 평가, 그리고 결과 시각화를 포함한 전반적인 머신러닝 파이프라인을 구현함.

# 스크립트는 다음과 같은 작업을 수행함:
# 1. 필요한 라이브러리 및 패키지들을 임포트하여 모델 학습과 평가에 필요한 환경을 구성함. 여기에는 PyTorch, torchvision, transformers, EfficientNet, scikit-learn 등이 포함됨.
# 2. 운영 체제에 따라 경로를 설정하고, 데이터가 저장된 디렉토리를 지정함. 이로써 다양한 플랫폼에서의 호환성을 확보함.
# 3. 데이터를 로드하고 전처리하는 과정을 수행함. 데이터셋은 CSV 파일을 통해 불러오며, 이미지 데이터에 대해 필요한 변환(transform)을 적용하여 모델 학습에 적합한 형태로 변환함.
# 4. PyTorch의 Dataset 클래스를 확장하여 커스텀 데이터셋 클래스를 정의하고, DataLoader를 통해 배치 단위로 데이터를 로드할 수 있도록 함.
# 5. 다양한 딥러닝 모델을 초기화하고, 학습 파라미터(optimizer, learning rate 등)를 설정함. 여기에는 EfficientNet, ViT, ResNet, SENet, ResNet-SENet, 그리고 torchvision에서 제공하는 사전 학습된 모델들이 포함됨. 이때, argparse를 사용하여 명령줄에서 모델 유형, 학습률, 배치 크기 등의 파라미터를 유연하게 조정할 수 있게 함.
# 6. 학습 루프를 통해 모델을 학습시키고, 검증 데이터를 통해 주기적으로 모델 성능을 평가함. 이 과정에서 손실 함수(loss)를 계산하고, 백워드 패스(backward pass)를 통해 가중치를 갱신함.
# 7. 학습이 완료된 모델을 평가 데이터셋에서 테스트하고, F1 스코어와 같은 성능 지표를 계산하여 모델의 성능을 정량적으로 평가함.
# 8. 학습 및 평가 과정에서 발생한 손실과 정확도를 시각화하여, 모델의 학습 진행 상황을 그래프로 표현함.
# 9. 최종적으로 학습된 모델을 저장하고, 추후에 모델을 로드하여 재사용하거나, 추가적인 테스트를 수행할 수 있도록 함.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from transformers import ViTForImageClassification
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import ssl
import platform
import time
import argparse
import random
import numpy as np
from matplotlib import font_manager, rc

# Seed 설정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA, ensure all devices use the same seed
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility for convolutional layers
    torch.backends.cudnn.benchmark = False

# SEED 값 설정
SEED = 42
set_seed(SEED)

ssl._create_default_https_context = ssl._create_unverified_context

# 운영 체제에 따라 경로 설정
if platform.system() == "Windows":
    WORKING_DIR = r"D:\work"
else:  # macOS 또는 다른 유닉스 계열 운영체제
    WORKING_DIR = "/Users/a07874/work/trash_model"


TRAIN_DIR = os.path.join(WORKING_DIR, "train")
PREDICT_DIR = os.path.join(WORKING_DIR, "predict")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
PREDICT_CSV_DIR = os.path.join(WORKING_DIR,"OUTPUT")
TRAIN_CSV_PATH = os.path.join(WORKING_DIR, "csv")
TEST_CSV_PATH = os.path.join(WORKING_DIR, "csv","test_data.csv")


# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None,is_training = True):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_training = is_training
          # CSV 파일 읽기 (문제 해결을 위한 설정 추가)
        
        
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]

        # NaN 값이 있으면 처리
        if pd.isna(img_name) or img_name == 'nan':
            raise ValueError(f"Invalid image file name at index {idx}: {img_name}")

        img_path = os.path.join(self.image_dir, str(img_name))

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        if self.transform:
            image = self.transform(image)

        
        if self.is_training:
            label = int(self.data.iloc[idx, 1])  # 학습 데이터일 경우 label 사용
            return image, label
        else:
            return image
       
# 전처리 및 변환 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 랜덤 수평 뒤집기
    transforms.RandomRotation(10),  # 랜덤 회전  # 이미지 크기를 ViT에 맞춰 224x224로 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 전처리 및 변환 정의
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기를 ViT에 맞춰 224x224로 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=30, patience=2, checkpoint_path=None):
    best_f1 = 0.0
    best_model_wts = model.state_dict()
    start_epoch = 0

    train_f1_scores = []
    val_f1_scores = []

    # 조기 종료 로직을 위한 변수 초기화
    epochs_no_improve = 0
    early_stop = False
    
    # 이전 체크포인트가 있다면 로드
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        best_model_wts = checkpoint['best_model_wts']
        print(f"Resuming training from epoch {start_epoch}")

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        if early_stop:
            print("Early stopping")
            break

        print(f'Epoch {epoch+1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
               
            else:
                model.eval()
                dataloader = dataloaders['val']
                

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if hasattr(outputs, 'logits'):  # ViT 모델일 경우
                        outputs = outputs.logits
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')
            
                # F1 점수를 리스트에 추가
            if phase == 'train':
                train_f1_scores.append(epoch_f1)
            else:
                val_f1_scores.append(epoch_f1)

            # 검증 단계에서 성능이 개선되었는지 확인
            if phase == 'val':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = model.state_dict()
                    epochs_no_improve = 0  # 성능이 개선되면 카운터 리셋
                else:
                    epochs_no_improve += 1  # 성능이 개선되지 않으면 카운터 증가

                # patience만큼 에포크 동안 성능이 개선되지 않으면 학습 중단
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    early_stop = True
                    break

        # 체크포인트 저장
        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_model_wts': best_model_wts
            }, checkpoint_path)

    total_time = time.time() - start_time
    print(f'Total training time for {num_epochs} epochs: {total_time:.2f} seconds')

    print(f'Best val F1: {best_f1:.4f}')
    
    return best_f1, best_model_wts, total_time, train_f1_scores, val_f1_scores


# 예측 함수 정의
def predict_and_save(model, dataloader, output_csv, device):
    model.eval()
    all_preds = []

    start_time = time.time()

    for inputs in dataloader:
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            if hasattr(outputs, 'logits'):  # ViT 모델일 경우
                outputs = outputs.logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    total_time = time.time() - start_time
    print(f'Total inference time: {total_time:.2f} seconds')

    predict_df = pd.read_csv(output_csv)
    predict_df['category'] = all_preds
    predict_df.to_csv(output_csv, index=False)

    return total_time


# def create_weighted_sampler(train_csv_path):
#     # 클래스 카운트 계산
#     class_counts = pd.read_csv(train_csv_path)['category'].value_counts()
    
#     # 클래스 가중치 초기화
#     class_weights = {i: 1.0 for i in range(83)}  # 84개의 클래스를 위해 초기화
    
#     # 각 클래스의 가중치 계산
#     for cls in class_counts.index:
#         class_weights[cls] = 1.0 / class_counts[cls]  # 예시로 역수 가중치 사용
    
#     # 각 샘플에 대한 가중치 리스트 생성
#     weights = []
#     df = pd.read_csv(train_csv_path)
    
#     for category in df['category']:
#         weights.append(class_weights[category])
    
     
#     sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    
    
#     return sampler, torch.tensor(list(class_weights.values()), dtype=torch.float32)

# 메인 함수
def main(selected_model, epoch_count):
    # GPU 사용 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   

    # 학습 데이터 및 데이터로더 생성
    
    train_csv_path = os.path.join(TRAIN_CSV_PATH, "train_data.csv")
    train_dataset = CustomDataset(image_dir=TRAIN_DIR, csv_file=train_csv_path, transform=train_transform,is_training=True)
    
    # sampler, class_weights_tensor = create_weighted_sampler(train_csv_path)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,pin_memory=True,num_workers=4)

    # 정답 레이블이 포함된 테스트 데이터 로드
    test_data = pd.read_csv(TEST_CSV_PATH)

    # 예측용 데이터 로드 및 데이터로더 생성
    predict_csv_files = {
        'resnet': os.path.join(PREDICT_CSV_DIR, 'r_predict.csv'),
        'senet': os.path.join(PREDICT_CSV_DIR, 's_predict.csv'),
        'resnet-senet': os.path.join(PREDICT_CSV_DIR, 'rs_predict.csv'),
        'efficientnet-b7': os.path.join(PREDICT_CSV_DIR, 'e_predict.csv'),
        'vit': os.path.join(PREDICT_CSV_DIR, 'vit_predict.csv')
    }

    # 체크포인트 디렉터리 설정 및 생성
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # 디렉터리가 없으면 생성

    # 모델 선택
    models_to_train = {}
    NUM_CLASSES = 83

    if selected_model in [0, 1]:
        models_to_train['resnet'] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        models_to_train['resnet'].fc = nn.Linear(models_to_train['resnet'].fc.in_features, NUM_CLASSES)
    if selected_model in [0, 2]:
        models_to_train['senet'] = models.squeezenet1_0(pretrained=True)
        models_to_train['senet'].classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))  # SE-Net으로 대체
    if selected_model in [0, 3]:
        models_to_train['resnet-senet'] = models.resnet50(pretrained=True)  # ResNet-SE-Net 모델 정의 필요
        models_to_train['resnet-senet'].fc = nn.Linear(models_to_train['resnet-senet'].fc.in_features, NUM_CLASSES)
    if selected_model in [0, 4]:
        models_to_train['efficientnet-b7'] = EfficientNet.from_pretrained('efficientnet-b7')  # EfficientNet-B7
        models_to_train['efficientnet-b7']._fc = nn.Linear(models_to_train['efficientnet-b7']._fc.in_features, NUM_CLASSES)
    if selected_model in [0, 5]:
        models_to_train['vit'] = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)  # Vision Transformer

    best_overall_f1 = 0.0
    best_overall_wts = None
    f1_scores = {}
    training_times = {}
    inference_times = {}

    for model_name, model in models_to_train.items():
        print(f"Training {model_name}...")

        # 모델을 GPU로 이동
        model = model.to(device)

        # 손실 함수 및 최적화 방법 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        # 체크포인트 파일 경로 설정
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_checkpoint.pth')

          # 예측 수행 및 저장
    

        # 모델 학습 및 가장 좋은 가중치 저장
        best_f1, best_model_wts, training_time, train_f1_scores, val_f1_scores = train_model(
            model, 
            {'train': train_loader, 'val': train_loader}, 
            criterion, 
            optimizer, 
            device, 
            num_epochs=epoch_count, 
            checkpoint_path=checkpoint_path
        )

        # 모델 저장
        model_filename = os.path.join(WORKING_DIR, f"best_{model_name}.pth")
        torch.save(best_model_wts, model_filename)
        print(f"{model_name} model saved as {model_filename}")

        # 학습 시간 기록
        training_times[model_name] = training_time

        predict_dataset = CustomDataset(image_dir=PREDICT_DIR, csv_file=predict_csv_files[model_name], transform=predict_transform)
        predict_loader = DataLoader(predict_dataset, batch_size=64, shuffle=False,pin_memory=True,num_workers=4)
      
        
        inference_time = predict_and_save(model, predict_loader, predict_csv_files[model_name], device)

        # 예측 시간 기록
        inference_times[model_name] = inference_time

        # 예측 수행 및 저장 후
        predict_df = pd.read_csv(predict_csv_files[model_name])
        print("Predict DataFrame columns:", predict_df.columns)  # 디버깅용 출력
        merged_df = pd.merge(test_data, predict_df, on='file_name')
        print("Merged DataFrame columns:", merged_df.columns)  # 디버깅용 출력

        # 열 이름을 'category_x'와 'category_y'로 수정하여 F1 스코어 계산
        if 'category_x' in merged_df.columns and 'category_y' in merged_df.columns:
            f1 = f1_score(merged_df['category_x'], merged_df['category_y'], average='weighted')
            f1_scores[model_name] = f1
            print(f'{model_name} Prediction F1 Score: {f1:.4f}')

            # 가장 좋은 모델 찾기
            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_overall_wts = best_model_wts

            # `merged_df`를 CSV 파일로 저장
            merged_filename = os.path.join(WORKING_DIR, f'{model_name}_test.csv')
            merged_df.to_csv(merged_filename, index=False)
            print(f"{model_name} merged DataFrame saved as {merged_filename}")

        else:
            print(f"Error: 'category_x' or 'category_y' column is missing in the merged dataframe for model {model_name}. Skipping F1 score calculation.")
            continue

    # 전체 모델 중 가장 좋은 모델 저장
    if best_overall_wts is not None:
        torch.save(best_overall_wts, os.path.join(WORKING_DIR, 'best_model.pth'))
        print(f"The best model saved as best_model.pth with F1 score: {best_overall_f1:.4f}")
    else:
        print("No valid model was found.")

    # 학습 시간과 추론 시간을 출력
    print("\nModel Training Times (seconds):")
    for model_name, time_taken in training_times.items():
        print(f"{model_name}: {time_taken:.2f}")

    print("\nModel Inference Times (seconds):")
    for model_name, time_taken in inference_times.items():
        print(f"{model_name}: {time_taken:.2f}")

    # F1 점수를 그래프로 시각화
    if f1_scores:
        if platform.system() == "Windows":
            # 설치된 폰트 경로 확인
            font_path = "C:/Windows/Fonts/malgun.ttf"  # '맑은 고딕' 폰트 경로 예시

            # 폰트 등록
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)

            # 한글 깨짐 방지 설정
            plt.rcParams['axes.unicode_minus'] = False
        else:  # macOS 또는 다른 유닉스 계열 운영체제
            # AppleGothic 폰트 설정
            rc('font', family='AppleGothic')

            # 한글 깨짐 방지 설정
            plt.rcParams['axes.unicode_minus'] = False


        plt.figure(figsize=(10, 6))
        plt.bar(f1_scores.keys(), f1_scores.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
        plt.title('모델별 F1 점수')
        plt.xlabel('모델')
        plt.ylabel('F1 점수')
        plt.ylim(0, 1)
        plt.show()
    else:
        print("No F1 scores to display.")
    


    # 각 모델의 훈련 F1 점수 시각화
    for model_name, f1_list in zip(models_to_train.keys(), [train_f1_scores, val_f1_scores]):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch_count + 1), f1_list, marker='o', label=model_name)
        plt.title(f'{model_name}의 F1 점수 변화')
        plt.xlabel('Epoch')
        plt.ylabel('F1 점수')
        plt.ylim(0, 1)
        plt.grid()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    #실행 예시: python trash_final.py --selected_model 0 --epoch_count 5 모든 모델
    #실행 예시: python trash_final.py --selected_model 1 --epoch_count 5 resnet
    parser = argparse.ArgumentParser(description="Train and evaluate selected models.")
    parser.add_argument('--selected_model', type=int, default=0, help="Select model: 1=ResNet, 2=SENet, 3=ResNet-SENet, 4=EfficientNet-B7, 5=ViT, 0=All models")
    parser.add_argument('--epoch_count', type=int, default=5, help="Number of epochs to train the model(s)")

    args = parser.parse_args()
    main(args.selected_model, args.epoch_count)