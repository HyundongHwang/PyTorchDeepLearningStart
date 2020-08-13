import myutil as mu

################################################################################
# - 장단기 메모리(Long Short-Term Memory, LSTM)
# - 바닐라 RNN의 한계
#   - RNN은 출력 결과가 이전의 계산 결과에 의존한다는 것을 언급한 바 있습니다.
#   - 하지만 바닐라 RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있습니다.
#   - 바닐라 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생합니다.
#   - 위의 그림은 첫번째 입력값인 x1의 정보량을 짙은 남색으로 표현했을 때,
#   - 색이 점차 얕아지는 것으로 시점이 지날수록 x1의 정보량이 손실되어가는 과정을 표현하였습니다.
#   - 뒤로 갈수록 x1의 정보량은 손실되고,
#   - 시점이 충분히 긴 상황에서는 x1의 전체 정보에 대한 영향력은 거의 의미가 없을 수도 있습니다.
#
# ![](https://wikidocs.net/images/page/22888/lstm_image1_ver2.PNG)
#
#   - 어쩌면 가장 중요한 정보가 시점의 앞 쪽에 위치할 수도 있습니다.
#   - RNN으로 만든 언어 모델이 다음 단어를 예측하는 과정을 생각해봅시다.
#   - 예를 들어
#           - 모스크바에 여행을 왔는데 건물도 예쁘고 먹을 것도 맛있었어.
#           - 그런데 글쎄 직장 상사한테 전화가 왔어.
#           - 어디냐고 묻더라구 그래서 나는 말했지. 저 여행왔는데요. 여기 ___
#   - 다음 단어를 예측하기 위해서는 장소 정보가 필요합니다.
#   - 그런데 장소 정보에 해당되는 단어인 '모스크바'는 앞에 위치하고 있고,
#   - RNN이 충분한 기억력을 가지고 있지 못한다면 
#   - 다음 단어를 엉뚱하게 예측합니다.
#
#   - 이를 장기 의존성 문제(the problem of Long-Term Dependencies)라고 합니다.

################################################################################
#
#
# - 바닐라 RNN 내부 열어보기
#
# ![](https://wikidocs.net/images/page/22888/vanilla_rnn_ver2.PNG)


################################################################################
#
#
# - LSTM(Long Short-Term Memory)
#   - LSTM은 은닉층의 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여
#   - 불필요한 기억을 지우고, 기억해야할 것들을 정합니다.
#   - 요약하면 LSTM은 은닉 상태(hidden state)를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌으며
#   - 셀 상태(cell state)라는 값을 추가하였습니다.
#   - 위의 그림에서는 t시점의 셀 상태를 Ct로 표현하고 있습니다.
#   - LSTM은 RNN과 비교하여 긴 시퀀스의 입력을 처리하는데 탁월한 성능을 보입니다.
#
#
# ![](https://wikidocs.net/images/page/22888/vaniila_rnn_and_different_lstm_ver2.PNG)


import numpy as np
import torch
import torch.nn as nn

inputs = torch.Tensor(1, 10, 5)

cell = nn.LSTM(
    input_size=5,
    hidden_size=8,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

outputs, _status = cell(inputs)
mu.log("_status", _status)
mu.log("outputs.shape", outputs.shape)
