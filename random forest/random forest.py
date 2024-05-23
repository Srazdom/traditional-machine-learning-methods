import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# 加载数据
data = pd.read_csv("F:/intern in FSD/test for double integrator/data generate/generated_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 选择前两个特征进行可视化
X = X[:, :2]

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用随机森林模型进行训练
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"随机森林分类准确率: {accuracy_rf:.9f}")

# 使用交叉验证评估
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

print(f"随机森林交叉验证准确率: {evaluate_model(clf_rf, X, y):.9f}")

# 绘制决策边界
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1],alpha=0.6, c=y, s=20, edgecolor='k')
    plt.title(title, fontproperties="SimHei")  # 设置字体为支持中文的字体
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# 绘制随机森林模型的决策边界
plot_decision_boundary(clf_rf, X, y, f"随机森林 决策边界 (准确率: {accuracy_rf:.9f})")
