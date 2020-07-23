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

################################################################################
# 스택킹(Stacking)

x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
mu.log("torch.stack([x, y, z])", torch.stack([x, y, z]))

mu.log("torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0)",
       torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

mu.log("torch.stack([x, y, z], dim=1)", torch.stack([x, y, z], dim=1))

################################################################################
# ones_like와 zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
mu.log("x", x)
mu.log("torch.ones_like(x)", torch.ones_like(x))
mu.log("torch.zeros_like(x)", torch.zeros_like(x))

################################################################################
# In-place Operation (덮어쓰기 연산)

x = torch.FloatTensor([[1, 2], [3, 4]])

mu.log("x", x)
mu.log("x.mul(2.)", x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
mu.log("x", x)
mu.log("x.mul_(2.)", x.mul_(2.)) # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
mu.log("x", x)
