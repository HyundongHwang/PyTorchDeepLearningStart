import myutil as mu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더
from torch.utils.data import Dataset

################################################################################
# - 로지스틱 회귀(Logistic Regression)
#   - 일상 속 풀고자하는 많은 문제 중에서는 두 개의 선택지 중에서 정답을 고르는 문제가 많습니다.
#   - 예를 들어 시험을 봤는데 이 시험 점수가 합격인지 불합격인지가 궁금할 수도 있고, 어떤 메일을 받았을 때 이게 정상 메일인지 스팸 메일인지를 분류하는 문제도 그렇습니다. 이렇게 둘 중 하나를 결정하는 문제를 이진 분류(Binary Classification)라고 합니다. 그리고 이진 분류를 풀기 위한 대표적인 알고리즘으로 로지스틱 회귀(Logistic Regression)가 있습니다.
#   - 로지스틱 회귀는 알고리즘의 이름은 회귀이지만 실제로는 분류(Classification) 작업에 사용할 수 있습니다.


################################################################################
# - 시그모이드 함수(Sigmoid function)
#   - 위와 같이 S자 형태로 그래프를 그려주는 시그모이드 함수의 방정식은 아래와 같습니다.
#
# ```
# H(x) = sigmoid(Wx + b) = \frac{1}{1 + e^{-(Wx + b)}} = σ(Wx + b)
# ```
#
#   - 선형 회귀에서는 최적의 W와 b를 찾는 것이 목표였습니다.
#   - 여기서도 마찬가지입니다.
#   - 선형 회귀에서는 W가 직선의 기울기, b가 y절편을 의미했습니다.
#   - 그렇다면 여기에서는 W와 b가 함수의 그래프에 어떤 영향을 주는지 직접 그래프를 그려서 알아보겠습니다.


# %matplotlib inline
import numpy as np  # 넘파이 사용
import matplotlib.pyplot as plt  # 맷플롯립사용


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, "g")
plt.plot([0, 0], [1, 0], ":")
plt.title("sigmoid function")
plt.show()
