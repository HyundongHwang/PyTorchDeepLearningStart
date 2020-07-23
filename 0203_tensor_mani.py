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