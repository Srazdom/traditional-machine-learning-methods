import numpy as np
import pandas as pd

# 定义函数 h(x)
def h(x):
    x1, x2 = x
    if x2 > 0:
        return -2 * x1 - x2**2
    else:
        return -x1

# 生成随机数据
np.random.seed(0)
x1 = np.random.uniform(-2, 0.5, 3600)
x2 = np.random.uniform(-2, 2, 3600)
X = np.vstack((x1, x2)).T

# 依据 h(x) 生成标签
y = np.array([1 if h(point) >= 0 else 0 for point in X])

# 创建DataFrame
data = pd.DataFrame(X, columns=['x1', 'x2'])
data['safety'] = y
data.to_csv("F:/intern in FSD/test for double integrator/data generate/generated_data.csv", index=False)
