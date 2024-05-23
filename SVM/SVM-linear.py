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
X = data[['x1', 'x2']].values
y = data['safety'].values

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 遗传算法设置
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.1, 10.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalSVM(individual):
    C = max(0.01, individual[0])  # 确保 C 参数为正值
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return accuracy_score(y_test, predictions),

toolbox.register("evaluate", evalSVM)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=1.0, sigma=0.5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
pop = toolbox.population(n=30)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.05, ngen=15, verbose=True)

# 提取最佳个体
top_individual = tools.selBest(pop, k=1)[0]
print(f'Best C found by GA: {top_individual[0]}')

# 网格搜索
param_grid = {
    'C': np.linspace(max(0.01, top_individual[0] - 1), top_individual[0] + 1, 100)
}
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Best C found by Grid Search: {grid_search.best_params_["C"]}')
print(f'Best Score from Grid Search: {grid_search.best_score_}')

# 绘制热图
results = grid_search.cv_results_
scores = results['mean_test_score']
plt.figure(figsize=(10, 4))
sns.lineplot(x=param_grid['C'], y=scores, marker='o')
plt.title('SVM Linear Kernel Performance')
plt.xlabel('C parameter')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


