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
toolbox.register("attr_gamma", random.uniform, 0.0001, 5.0)
toolbox.register("attr_coef0", random.uniform, -5, 5)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_c, toolbox.attr_gamma, toolbox.attr_coef0), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalSVM(ind):
    C, gamma, coef0 = ind
    clf = SVC(kernel='sigmoid', C=C, gamma=gamma, coef0=coef0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return (accuracy_score(y_test, predictions),)

toolbox.register("evaluate", evalSVM)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0.1, 0.0001, -5], up=[10, 5, 5], indpb=0.1, eta=20)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.05, ngen=20, verbose=True)

# 提取最佳个体
top_individual = tools.selBest(pop, k=1)[0]
print(f'Best parameters found by GA: C={top_individual[0]}, gamma={top_individual[1]}, coef0={top_individual[2]}')

# 网格搜索以优化参数
param_grid = {
    'C': np.linspace(max(0.01, top_individual[0] - 2), top_individual[0] + 2, 10),
    'gamma': np.linspace(max(0.0001, top_individual[1] - 0.5), top_individual[1] + 0.5, 10),
    'coef0': np.linspace(top_individual[2] - 1, top_individual[2] + 1, 10)
}

grid_search = GridSearchCV(SVC(kernel='sigmoid'), param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
print(f'Best parameters found by Grid Search: {grid_search.best_params_}')
print(f'Best Score from Grid Search: {grid_search.best_score_}')

# 可视化网格搜索结果
results = grid_search.cv_results_
n_C = len(param_grid['C'])
n_gamma = len(param_grid['gamma'])
n_coef0 = len(param_grid['coef0'])
scores = results['mean_test_score'].reshape(n_C, n_gamma, n_coef0)

# 绘制每个 gamma 对应的热图
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10), constrained_layout=True)
for i, gamma in enumerate(param_grid['gamma']):
    ax = axes.flat[i]
    sns.heatmap(scores[:, i, :], annot=True, fmt=".2f", xticklabels=np.round(param_grid['coef0'], 2), yticklabels=np.round(param_grid['C'], 2), cmap='coolwarm', ax=ax)
    ax.set_title(f'gamma={gamma:.4f}')
    ax.set_xlabel('Coef0')
    ax.set_ylabel('C')

plt.suptitle('SVM Sigmoid Kernel Performance for Different Gamma Values', fontsize=16)
plt.show()




