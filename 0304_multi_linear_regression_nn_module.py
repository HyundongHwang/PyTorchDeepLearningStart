################################################################################
# 다중 선형 회귀(Multivariable Linear regression)

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
mu.log("list(model.parameters())", list(model.parameters()))
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
