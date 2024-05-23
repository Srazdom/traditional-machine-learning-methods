import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import random

# 加载数据
data_path = "F:/intern in FSD/test for double integrator/data generate/generated_data.csv"
data = pd.read_csv(data_path)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 遗传算法设置
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_c", random.uniform, 0.1, 10.0)
toolbox.register("attr_degree", random.randint, 1, 5)
toolbox.register("attr_coef0", random.uniform, 0, 5)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_c, toolbox.attr_degree, toolbox.attr_coef0), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalSVM(ind):
    C, degree, coef0 = ind
    clf = SVC(kernel='poly', C=C, degree=int(degree), coef0=coef0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return accuracy_score(y_test, predictions),

toolbox.register("evaluate", evalSVM)
toolbox.register("mate", tools.cxTwoPoint)

# 自定义变异以确保整数度
def custom_mutate(individual, low, up, indpb, eta):
    size = len(individual)
    for i, lo, up in zip(range(size), low, up):
        if random.random() < indpb:
            x = individual[i]
            if i == 1:  # 确保 'degree' 保持为整数
                x = int(x)
                x += random.randint(-1, 1)
                x = max(lo, min(x, up))  # 约束变异
                individual[i] = x
            else:
                xq = (x - lo) / (up - lo)
                xq += random.uniform(-0.05, 0.05)  # 小幅变异
                xq = max(0, min(1, xq))
                x = lo + xq * (up - lo)
                individual[i] = x
    return individual,

toolbox.register("mutate", custom_mutate, low=[0.1, 1, 0], up=[10, 5, 5], indpb=0.1, eta=20)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
pop = toolbox.population(n=30)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.05, ngen=15, verbose=True)

# 提取最佳个体
top_individual = tools.selBest(pop, k=1)[0]
print(f'Best parameters found by GA: C={top_individual[0]}, degree={top_individual[1]}, coef0={top_individual[2]}')

# 网格搜索以优化参数
param_grid = {
    'C': np.linspace(max(0.01, top_individual[0] - 2), top_individual[0] + 2, 10),
    'degree': range(max(1, top_individual[1] - 1), top_individual[1] + 2),
    'coef0': np.linspace(max(0, top_individual[2] - 1), top_individual[2] + 1, 10)
}

grid_search = GridSearchCV(SVC(kernel='poly'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Best parameters found by Grid Search: {grid_search.best_params_}')
print(f'Best Score from Grid Search: {grid_search.best_score_}')

# 绘制热图
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']

# 调整形状
scores = mean_test_scores.reshape(len(param_grid['C']), len(param_grid['degree']), len(param_grid['coef0']))

# 绘制每个 degree 对应的热图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
for i, degree in enumerate(param_grid['degree']):
    ax = axes[i]
    sns.heatmap(scores[:, i, :], annot=True, fmt=".3f", xticklabels=np.round(param_grid['coef0'], 2), yticklabels=np.round(param_grid['C'], 2), cmap='viridis', ax=ax)
    ax.set_title(f'SVM Polynomial Kernel Performance (degree={degree})')
    ax.set_xlabel('coef0')
    ax.set_ylabel('C')

plt.tight_layout()
plt.show()

