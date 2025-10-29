# import numpy as np
# from scipy import stats

# np.random.seed(0)

# for i in range(10):
#     res = stats.bernoulli(0.5).rvs(10)
#     print(np.sum(res))

import torch
import torch.nn.functional as F

# 创建一个示例输入
x = torch.randn(1, 3, 4, 4)  # (batch_size, channels, height, width)

# 创建一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        x = F.dropout2d(x, p=0.5, training=self.training)
        return x

model = SimpleModel()

print("原始输入")
print(x)


# 设置模型为训练模式
model.train()
output_train = model(x)
print("Output during training:")
print(output_train)

# 设置模型为推理模式
model.eval()
output_eval = model(x)
print("Output during evaluation:")
print(output_eval)
