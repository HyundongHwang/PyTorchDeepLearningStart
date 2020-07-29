import myutil as mu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더
from torch.utils.data import Dataset
import matplotlib.pyplot as plt  # 맷플롯립사용

################################################################################
# - 파이토치로 소프트맥스의 비용 함수 구현하기 (로우-레벨)

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
mu.log("z", z)
mu.log("hypothesis", hypothesis)
mu.log("hypothesis.sum()", hypothesis.sum())

z = torch.rand(3, 5, requires_grad=True)
mu.log("z", z)
hypothesis = F.softmax(z, dim=1)
mu.log("hypothesis", hypothesis)
mu.log("hypothesis.sum(dim=1)", hypothesis.sum(dim=1))

y = torch.randint(5, (3,))
mu.log("y", y)
hypothesis = F.softmax(z, dim=1)
mu.log("hypothesis", hypothesis)
y_one_hot = torch.zeros_like(hypothesis)
mu.log("y_one_hot", y_one_hot)
mu.log("y.unsqueeze(1)", y.unsqueeze(1))
mu.log("y_one_hot.scatter_(dim=1, y.unsqueeze(dim=1), index=1)", y_one_hot.scatter_(1, y.unsqueeze(1), 1))

################################################################################
# - 이제 비용 함수 연산을 위한 재료들을 전부 손질했습니다.
# - 소프트맥스 회귀의 비용 함수는 다음과 같았습니다.
# ![](https://render.githubusercontent.com/render/math?math=cost(W)%20=%20-\frac{1}{n}%20\sum_{i=1}^{n}%20\sum_{j=1}^{k}y_{j}^{(i)}\%20log(p_{j}^{(i)}))

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
mu.log("cost", cost)

################################################################################
# - 파이토치로 소프트맥스의 비용 함수 구현하기 (하이-레벨)
# - 이제 소프트맥스의 비용 함수를 좀 더 하이-레벨로 구현하는 방법에 대해서 알아봅시다.

z = torch.rand(3, 5, requires_grad=True)
mu.log("z", z)
y = torch.randint(5, (3,))
mu.log("y", y)
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
mu.log("y_one_hot", y_one_hot)
hypothesis = F.softmax(z, dim=1)
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
mu.log("cost low level", cost)

################################################################################
# - F.softmax() + torch.log() = F.log_softmax()

cost = (y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()
mu.log("cost log_softmax", cost)

################################################################################
# - 여기서 nll이란 Negative Log Likelihood의 약자입니다.

cost = F.nll_loss(F.log_softmax(z, dim=1), y)
mu.log("cost nll_loss log_softmax", cost)

################################################################################
# - F.log_softmax() + F.nll_loss() = F.cross_entropy()

cost = F.cross_entropy(z, y)
mu.log("cost cross_entropy", cost)
