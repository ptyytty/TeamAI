import torch # pytorch default 라이브러리
import torch.nn as nn # 신경망 구축을 위한 모듈
import torch.optim as optim # 최적화 알고리즘 제공
import torchvision # dataset(mnist 등) 모듈 제공
import torchvision.transforms as transforms # 데이터 전처리 위한 모듈 import
print(torch.__version__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(777)

if device == "cuda":
    torch.cuda.manual_seed(777)


# 파라미터 정의
learning_rate = 0.001 # 학습률
training_epochs = 15 # 데이터셋을 몇번 반복할 횟수 = 에포크 수
batch_size = 100 # 처리할 데이터 샘플 수 

# mnist 데이터 셋 로드  + 데이터 전처리
# .compose () => 여러개의 전처리 작업을 순차적으로 적용할 수 있도록 묶는 함수
# .Totenser() => 이미지 텐서로 변환
# .Normalize() => 이미지 정규화 (평균, 표준편차)
transform = transforms.Compose([
    transforms.ToTensor(), # 텐서 변환
    transforms.Normalize((0.5,),(0.5,)) # avg = 0.5 SD = 0.5
])

#학습용 데이터 셋 로드
mnist_train = torchvision.datasets.MNIST(root='./data', # 다운로드 경로 지정
                          train= True,  # true 지정시 훈련 데이터로 다운로드 false 지정시 테스트 데이터로
                          transform= transform, # 적용할 전처리 파이프라인
                            download= True) # 데이터 셋 없을 경우 다운로드 여부
#테스트용 데이터 셋 로드
mnist_test = torchvision.datasets.MNIST(root='./data',
                          train= False,  
                          transform= transform,
                            download= True) 
#학습 데이터 로더
train_data_loader = torch.utils.data.DataLoader(dataset= mnist_train,
                                          batch_size= batch_size,
                                          shuffle= True)
                                          
#테스트 데이터 로더
test_data_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size= batch_size,
                                          shuffle= False)

#cnn 모델 정의
# nn.Module 상속
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__() # 초기화
        # 합성곱 층
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,padding= 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 전결합 층
        self.fc = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000,10)
    
    def forward(self, x) :
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1 ) # Flatten 전결합층 위해
        out = self.fc(out)
        out = self.fc2(out)
        return out
    

# 모델, 손실함수 및 옵티마이저 정의    
model = CNN().to(device)  # cnn 클래스의 인스턴스화
Criterion = nn.CrossEntropyLoss().to(device) #  분류를 위한 교차 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters() , lr=learning_rate) # 옵티마이저 정의(모델 모든 매개변수, 학습률)


# 모델 training

for epoch in range(training_epochs): # 에포크 수만큼 반복
    for i,(images, labels) in enumerate(train_data_loader):
        outputs = model(images)
        loss = Criterion(outputs,labels) # 손실 계산

        optimizer.zero_grad() # 옵티마이저의 변화 초기화
        loss.backward() # 손실을 역전파 하여 변화도계산
        optimizer.step() # 옵티마이저 사용하여 매개변수 업데이트


        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{training_epochs}], Step [{i+1}/{len(train_data_loader)}], Loss: {loss.item():.4f}')



# 모델 평가
model.eval()  # 평가 모드로 전환 (dropout, batchnorm 등의 동작을 멈춤)
with torch.no_grad():  # 평가 중에는 기울기를 계산하지 않음
    correct = 0
    total = 0
    for images, labels in test_data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')