import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv("F:/intern in FSD/test for double integrator/generated_data.csv")

# 分离特征和目标变量
X = data[['x1', 'x2']]
y = data['safety']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 网格搜索
# 使用之前遗传算法确定的最佳区间
C_range = np.logspace(np.log10(10), np.log10(100), 1000)
gamma_range = np.logspace(np.log10(0.1), np.log10(1), 1000)

param_grid = {
    'C': C_range,
    'gamma': gamma_range,
}

# 初始化 SVM 模型
svc = SVC(kernel='rbf')

# 初始化 GridSearchCV 对象，这里使用 5 折交叉验证
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=2)

# 对数据进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数组合和对应的准确率
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validated score: {best_score}")

# 可视化热图
# 提取测试得分
scores = grid_search.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

# 绘制热图
plt.figure(figsize=(10, 8))
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           extent=[np.log10(gamma_range).min(), np.log10(gamma_range).max(), 
                   np.log10(C_range).min(), np.log10(C_range).max()], origin='lower')
plt.colorbar()
plt.xlabel('log10(gamma)')
plt.ylabel('log10(C)')
plt.title('Validation accuracy as a function of C and gamma')
plt.xticks(np.round(np.log10(gamma_range), 2))
plt.yticks(np.round(np.log10(C_range), 2))
plt.show()
