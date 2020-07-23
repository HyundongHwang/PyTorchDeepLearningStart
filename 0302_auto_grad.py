import myutil as mu
import torch

################################################################################
# - 자동 미분(Autograd) 실습하기
#   - 자동 미분에 대해서 실습을 통해 이해해봅시다.
#   - 임의로 2w2+5라는 식을 세워보고, w에 대해 미분해보겠습니다.

w = torch.tensor(2.0, requires_grad=True)
y = w ** 2
z = 2 * y + 5
z.backward()
print('수식을 w로 미분한 값 : {}'.format(w.grad))
