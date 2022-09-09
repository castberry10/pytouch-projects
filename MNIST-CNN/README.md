# MNIST-CNN
mnist 0 ~ 9 분류 문제 

# CNN 모델
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (dropout1): Dropout2d(p=0.25, inplace=False)
  (dropout2): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)

실행 결과 

![image](https://user-images.githubusercontent.com/25453543/189338894-c6b82a07-9268-4625-9b6e-e300b8747c55.png)
