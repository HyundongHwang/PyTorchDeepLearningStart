import myutil as mu

################################################################################
# - 순환 신경망(Recurrent Neural Network, RNN)
#   - RNN(Recurrent Neural Network)은 시퀀스(Sequence) 모델입니다.
#   - 입력과 출력을 시퀀스 단위로 처리하는 모델입니다.
#   - 번역기를 생각해보면 입력은 번역하고자 하는 문장.
#   - 즉, 단어 시퀀스입니다.
#   - 출력에 해당되는 번역된 문장 또한 단어 시퀀스입니다.
#   - 이러한 시퀀스들을 처리하기 위해 고안된 모델들을 시퀀스 모델이라고 합니다.
#   - 그 중에서도 RNN은 딥 러닝에 있어 가장 기본적인 시퀀스 모델입니다.

################################################################################
# ![](https://wikidocs.net/images/page/22886/rnn_image3_ver2.PNG)

################################################################################
# ![](https://wikidocs.net/images/page/22886/rnn_image3.5.PNG)

################################################################################
# ![](https://wikidocs.net/images/page/22886/rnn_image3.7.PNG)


################################################################################
# ![](https://wikidocs.net/images/page/22886/rnn_image4_ver2.PNG)
#
# - 이를 식으로 표현하면 다음과 같습니다.
#   - 은닉층 : ht=tanh(Wxxt+Whht−1+b)
#   - 출력층 : yt=f(Wyht+b)
#   - 단, f는 비선형 활성화 함수 중 하나.
#
# xt  : (d×1)
# Wx : (Dh×d)
# Wh : (Dh×Dh)
# ht−1 : (Dh×1)
# b : (Dh×1)
#
# ![](https://wikidocs.net/images/page/22886/rnn_images4-5.PNG)

################################################################################
# - 파이썬으로 RNN 구현하기
#   - 직접 Numpy로 RNN 층을 구현해보겠습니다. 앞서 메모리 셀에서 은닉 상태를 계산하는 식을 다음과 같이 정의하였습니다.
#   - ht=tanh(WxXt+Whht−1+b)


import numpy as np

timesteps = 10
input_size = 4
hidden_size = 8

inputs = np.random.random((timesteps, input_size))
mu.log("inputs.shape", inputs.shape)

hidden_state_t = np.zeros((hidden_size,))
mu.log("hidden_state_t.shape", hidden_state_t.shape)

Wx = np.random.random((hidden_size, input_size))
Wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))

mu.log("Wx.shape", Wx.shape)
mu.log("Wh.shape", Wh.shape)
mu.log("b.shape", b.shape)

total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs:  # 각 시점에 따라서 입력값이 입력됨.
    print("-" * 80)
    mu.log("input_t.shape", input_t.shape)
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)  # Wx * Xt + Wh * Ht-1 + b(bias)
    mu.log("output_t.shape", output_t.shape)
    total_hidden_states.append(list(output_t))  # 각 시점의 은닉 상태의 값을 계속해서 축적
    mu.log("np.shape(total_hidden_states)",
           np.shape(total_hidden_states))  # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)
# 출력 시 값을 깔끔하게 해준다.

# (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.
mu.log("total_hidden_states", total_hidden_states)

################################################################################
# - 파이토치의 nn.RNN()
#   - 파이토치에서는 nn.RNN()을 통해서 RNN 셀을 구현합니다.

import torch
import torch.nn as nn

input_size = 5
hidden_size = 8
inputs = torch.Tensor(1, 10, 5)
mu.log("inputs", inputs)

cell = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    batch_first=True
)

outputs, _status = cell(inputs)
mu.log("_status.shape", _status.shape)
mu.log("outputs.shape", outputs.shape)

################################################################################
# - 깊은 순환 신경망(Deep Recurrent Neural Network)
#   - 앞서 RNN도 다수의 은닉층을 가질 수 있다고 언급한 바 있습니다.
#   - 위의 그림은 순환 신경망에서 은닉층이 1개 더 추가되어
#   - 은닉층이 2개인 깊은(deep) 순환 신경망의 모습을 보여줍니다.
#   - 위의 코드에서 첫번째 은닉층은 다음 은닉층에 모든 시점에 대해서 은닉 상태 값을 다음 은닉층으로 보내주고 있습니다.
#   - 깊은 순환 신경망을 파이토치로 구현할 때는 nn.RNN()의 인자인 num_layers에 값을 전달하여 층을 쌓습니다.
#   - 층이 2개인 깊은 순환 신경망의 경우,
#   - 앞서 실습했던 임의의 입력에 대해서 출력이 어떻게 달라지는지 확인해봅시다.
# ![](https://wikidocs.net/images/page/22886/rnn_image4.5_finalPNG.PNG)

input = torch.Tensor(1, 10, 5)

cell = nn.RNN(
    input_size=5,
    hidden_size=8,
    num_layers=2,
    batch_first=True
)

outputs, _status = cell(inputs)
mu.log("_status.shape", _status.shape)
mu.log("outputs.shape", outputs.shape)

################################################################################
# - 양방향 순환 신경망(Bidirectional Recurrent Neural Network)
#   - 양방향 순환 신경망은 시점 t에서의 출력값을 예측할 때 이전 시점의 데이터뿐만 아니라,
#   - 이후 데이터로도 예측할 수 있다는 아이디어에 기반합니다.
#
#   - 영어 빈칸 채우기 문제에 비유하여 보겠습니다.
#
# ```
# Exercise is very effective at [          ] belly fat.
#
# 1) reducing
# 2) increasing
# 3) multiplying
# ```
#
# - '운동은 복부 지방을 [ ] 효과적이다'라는 영어 문장이고,
# - 정답은 reducing(줄이는 것)입니다.
# - 그런데 위의 영어 빈 칸 채우기 문제를 잘 생각해보면
# - 정답을 찾기 위해서는 이전에 나온 단어들만으로는 부족합니다.
# - 목적어인 belly fat(복부 지방)를 모르는 상태라면 정답을 결정하기가 어렵습니다.
#
# - 그래서 이전 시점의 데이터뿐만 아니라,
# - 이후 시점의 데이터도 힌트로 활용하기 위해서 고안된 것이 양방향 RNN입니다.
#
# ![](https://wikidocs.net/images/page/22886/rnn_image6_ver3.PNG)

# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(
    input_size=5,
    hidden_size=8,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

outputs, _status = cell(inputs)

# (층의 개수 x 2, 배치 크기, 은닉 상태의 크기)
mu.log("_status.shape", _status.shape)

# (배치 크기, 시퀀스 길이, 은닉 상태의 크기 x 2)
mu.log("outputs.shape", outputs.shape)
