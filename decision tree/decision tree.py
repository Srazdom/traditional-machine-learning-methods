import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt

# 自定义 C4.5 实现
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
class DecisionTreeC45:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feature_idx = self._best_feature(X, y)
        if feature_idx is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)
        
        thresholds = np.unique(X[:, feature_idx])
        if len(thresholds) == 1:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)
        
        best_threshold = thresholds[0]
        left_idxs, right_idxs = X[:, feature_idx] <= best_threshold, X[:, feature_idx] > best_threshold
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return TreeNode(feature_index=feature_idx, threshold=best_threshold, left=left, right=right)

    def _best_feature(self, X, y):
        feature_gains = [self._information_gain_ratio(X, y, i) for i in range(X.shape[1])]
        return np.argmax(feature_gains)

    def _information_gain(self, X, y, feature_index):
        total_entropy = self._entropy(y)
        values, counts = np.unique(X[:, feature_index], return_counts=True)
        weighted_entropy = sum((counts[i] / len(y)) * self._entropy(y[X[:, feature_index] == value])
                               for i, value in enumerate(values))
        return total_entropy - weighted_entropy

    def _information_gain_ratio(self, X, y, feature_index):
        gain = self._information_gain(X, y, feature_index)
        split_info = self._entropy(X[:, feature_index])
        return gain / split_info if split_info != 0 else 0

    def _entropy(self, y):
        value, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        value, counts = np.unique(y, return_counts=True)
        return value[np.argmax(counts)]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {"min_samples_split": self.min_samples_split, "max_depth": self.max_depth}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# 加载数据
data = pd.read_csv("F:/intern in FSD/test for double integrator/data generate/generated_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 比较 CART, ID3, 和自定义 C4.5 的性能
# CART
clf_cart = DecisionTreeClassifier()
clf_cart.fit(X_train, y_train)
y_pred_cart = clf_cart.predict(X_test)
accuracy_cart = accuracy_score(y_test, y_pred_cart)
print(f"CART 分类准确率: {accuracy_cart:.9f}")

# ID3
clf_id3 = DecisionTreeClassifier(criterion='entropy')
clf_id3.fit(X_train, y_train)
y_pred_id3 = clf_id3.predict(X_test)
accuracy_id3 = accuracy_score(y_test, y_pred_id3)
print(f"ID3 分类准确率: {accuracy_id3:.9f}")

# 自定义 C4.5
clf_c45 = DecisionTreeC45()
clf_c45.fit(X_train, y_train)
y_pred_c45 = clf_c45.predict(X_test)
accuracy_c45 = accuracy_score(y_test, y_pred_c45)
print(f"自定义 C4.5 分类准确率: {accuracy_c45:.9f}")

# 使用交叉验证评估
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

print(f"CART 交叉验证准确率: {evaluate_model(clf_cart, X, y):.9f}")
print(f"ID3 交叉验证准确率: {evaluate_model(clf_id3, X, y):.9f}")
print(f"自定义 C4.5 交叉验证准确率: {evaluate_model(clf_c45, X, y):.9f}")

# 选择具有最高准确率的模型
best_model = None
best_accuracy = 0
best_model_name = ""

if accuracy_cart > best_accuracy:
    best_accuracy = accuracy_cart
    best_model = clf_cart
    best_model_name = "CART"

if accuracy_id3 > best_accuracy:
    best_accuracy = accuracy_id3
    best_model = clf_id3
    best_model_name = "ID3"

if accuracy_c45 > best_accuracy:
    best_accuracy = accuracy_c45
    best_model = clf_c45
    best_model_name = "自定义 C4.5"

# 绘制决策边界
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6,c=y, s=20, edgecolor='k')
    plt.title(title, fontproperties="SimHei")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# 绘制最佳模型的决策边界
plot_decision_boundary(best_model, X, y, f"{best_model_name} 决策边界 (准确率: {best_accuracy:.9f})")



