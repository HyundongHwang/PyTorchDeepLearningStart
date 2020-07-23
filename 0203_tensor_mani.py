import myutil as mu
import numpy as np
import torch

################################################################################
# 뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경. 매우 중요함!!

t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])

ft = torch.FloatTensor(t)
mu.log("t.shape", t.shape)
mu.log("ft.shape", ft.shape)

################################################################################
# 3차원 텐서에서 2차원 텐서로 변경

mu.log("ft.view([-1, 3])", ft.view([-1, 3]))
mu.log("ft.view([-1, 3]).shape", ft.view([-1, 3]).shape)

################################################################################
# 3차원 텐서의 크기 변경
mu.log("ft.view([-1, 1, 3])", ft.view([-1, 1, 3]))
mu.log("ft.view([-1, 1, 3]).shape", ft.view([-1, 1, 3]).shape)

################################################################################
# 스퀴즈(Squeeze) - 1인 차원을 제거한다.
ft = torch.FloatTensor([[0], [1], [2]])
mu.log("ft", ft)
mu.log("ft.shape", ft.shape)

################################################################################
# 언스퀴즈(Unsqueeze) - 특정 위치에 1인 차원을 추가한다.
ft = torch.Tensor([0, 1, 2])
mu.log("ft", ft)
mu.log("ft.shape", ft.shape)

mu.log("ft.unsqueeze(0)", ft.unsqueeze(0))  # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
mu.log("ft.unsqueeze(0).shape", ft.unsqueeze(0).shape)

mu.log("ft.view(1, -1)", ft.view(1, -1))
mu.log("ft.view(1, -1).shape", ft.view(1, -1).shape)

mu.log("ft.unsqueeze(1)", ft.unsqueeze(1))
mu.log("ft.unsqueeze(1).shape", ft.unsqueeze(1).shape)

mu.log("ft.unsqueeze(-1)", ft.unsqueeze(-1))
mu.log("ft.unsqueeze(-1).shape", ft.unsqueeze(-1).shape)

################################################################################
# 타입 캐스팅(Type Casting)

lt = torch.LongTensor([1, 2, 3, 4])
mu.log("lt", lt)
mu.log("lt.float()", lt.float())

bt = torch.ByteTensor([True, False, False, True])
mu.log("bt", bt)
mu.log("bt.long()", bt.long())
mu.log("bt.float()", bt.float())


################################################################################
# 연결하기(concatenate)

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
mu.log("torch.cat([x, y], dim=0)", torch.cat([x, y], dim=0))
mu.log("torch.cat([x, y], dim=1)", torch.cat([x, y], dim=1))
