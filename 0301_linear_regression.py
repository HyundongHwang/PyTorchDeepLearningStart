import myutil as mu
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

################################################################################
# 선형 회귀(Linear Regression)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000  # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        mu.log_epoch(epoch, nb_epochs, cost)
        mu.log("W", W)
        mu.log("b", b)

################################################################################
# optimizer.zero_grad()가 필요한 이유

import torch

w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    z = 2 * w

    z.backward()
    print('수식을 w로 미분한 값 : {}'.format(w.grad))

################################################################################
# torch.manual_seed()를 하는 이유

import torch

torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1, 3):
    print(torch.rand(1))

torch.manual_seed(5)
print('랜덤 시드가 5일 때')
for i in range(1, 3):
    print(torch.rand(1))

torch.manual_seed(3)
print('랜덤 시드가 다시 3일 때')
for i in range(1, 3):
    print(torch.rand(1))
