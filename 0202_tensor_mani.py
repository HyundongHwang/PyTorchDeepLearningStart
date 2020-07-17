import myutil as mu
import numpy as np
import torch

################################################################################
# 넘파이로 텐서 만들기(벡터와 행렬 만들기)

t = np.array([0., 1., 2., 3., 4., 5., 6.])
mu.log("t", t)
mu.log("t.ndim", t.ndim)
mu.log("t.shape", t.shape)
mu.log("t[-1]", t[-1])
mu.log("t[2:5]", t[2:5])
mu.log("t[4:-1]", t[4:-1])
mu.log("t[:2]", t[:2])
mu.log("t[3:]", t[3:])

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
mu.log("t", t)
mu.log("t.ndim", t.ndim)
mu.log("t.shape", t.shape)

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
mu.log("t", t)
mu.log("t.ndim", t.ndim)
mu.log("t.shape", t.shape)

################################################################################
# 파이토치 텐서 선언하기(PyTorch Tensor Allocation)

t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                       ])
mu.log("t", t)
mu.log("t.ndim", t.ndim)
mu.log("t.shape", t.shape)

################################################################################
# 행렬 곱셈과 곱셈의 차이(Matrix Multiplication Vs. Multiplication)

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
mu.log("m1", m1)
mu.log("m2", m2)
mu.log("m1 * m2", m1 * m2)
mu.log("m1.matmul(m2)", m1.matmul(m2))

################################################################################
# 평균(Mean)

t = torch.FloatTensor([1, 2])
mu.log("t.mean()", t.mean())

t = torch.FloatTensor([[1, 2], [3, 4]])
mu.log("t", t)
mu.log("t.mean(dim=0)", t.mean(dim=0))
mu.log("t.mean(dim=1)", t.mean(dim=1))

################################################################################
# 최대(Max)와 아그맥스(ArgMax)
t = torch.FloatTensor([[1, 2], [3, 4]])
mu.log("t", t)
mu.log("t.max()", t.max())
mu.log("t.max(dim=0)", t.max(dim=0))
mu.log("t.max(dim=1)", t.max(dim=1))
mu.log("t.max(dim=-1)", t.max(dim=-1))
