################################################################################
# - 다중 선형 회귀
#   - 이제 데이터를 선언합니다.
#   - 아래 데이터는 y=2x를 가정된 상태에서 만들어진 데이터로 우리는 이미 정답이 W=2, b=0임을 알고 있는 사태입니다.
#   - 모델이 이 두 W와 b의 값을 제대로 찾아내도록 하는 것이 목표입니다.

import myutil as mu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1, 1)
mu.log_model("model", model)
optimizer = optim.SGD(model.parameters(), lr=0.01)
nb_epochs = 2000

for epoch in range(nb_epochs + 1):
    hyperthesis = model(x_train)
    cost = F.mse_loss(hyperthesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("epoch {:4d}/{} cost : {:.6f} model : {}".format(
            epoch,
            nb_epochs,
            cost.item(),
            mu.model_to_str(model)
        ))

mu.log_model("model", model)
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
mu.log("new_var.item()", new_var.item())
mu.log("pred_y.item()", pred_y.item())
