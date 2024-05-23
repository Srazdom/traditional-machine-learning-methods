import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# 加载数据
data = pd.read_csv("F:/intern in FSD/test for double integrator/data generate/generated_data.csv")

# 分离特征和目标变量
X = data[['x1', 'x2']]
y = data['safety']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 定义适应度函数
def evalSVM(individual):
    C, gamma = individual
    try:
        clf = SVC(C=10**C, gamma=10**gamma, kernel='rbf')
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        return (scores.mean(),)
    except Exception as e:
        print(f"评估个体时发生错误: {e}")
        return (0,)  # 返回最小可能的准确率

# 检查是否已经创建了所需的类型，如果没有，则创建它们
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -4, 4)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalSVM)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
population = toolbox.population(n=30)
ngen = 15
CXPB, MUTPB = 0.7, 0.05

# 创建统计对象和日志
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)  # 只记录最大准确率

final_pop, logbook = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=ngen, stats=stats, verbose=True)

# 分析结果，找到最佳个体
best_ind = tools.selBest(final_pop, 1)[0]
best_C = 10 ** best_ind[0]
best_gamma = 10 ** best_ind[1]
best_accuracy = best_ind.fitness.values[0]
print(f"Best individual: C = {best_C}, gamma = {best_gamma}, Accuracy = {best_accuracy}")

# 可视化最大准确率
gen = logbook.select("gen")
max_accuracy = logbook.select("max")  # 只记录最大准确率

plt.figure(figsize=(11, 4))
plt.plot(gen, max_accuracy, label='Max Accuracy')
plt.xlabel('Generation')
plt.ylabel('Max Accuracy per Generation')
plt.title('Max Accuracy over Generations')
plt.legend()
plt.show()

# 网格搜索
# 使用之前遗传算法确定的最佳区间
C_range = np.logspace(np.log10(best_C * 0.1), np.log10(best_C * 10), 100)
gamma_range = np.logspace(np.log10(best_gamma * 0.1), np.log10(best_gamma * 10), 100)

param_grid = {
    'C': C_range,
    'gamma': gamma_range,
}

# 初始化 SVC 模型
svc = SVC(kernel='rbf')

# 初始化 GridSearchCV 对象，这里使用 5 折交叉验证，并增加 verbose 参数以显示进度
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=3)

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
plt.figure(figsize=(12, 10))
plt.imshow(scores, aspect='auto', interpolation='nearest', cmap='coolwarm',
           extent=[np.log10(gamma_range).min(), np.log10(gamma_range).max(), 
                   np.log10(C_range).min(), np.log10(C_range).max()], origin='lower')
plt.colorbar(label='Accuracy')
plt.xlabel('log10(gamma)')
plt.ylabel('log10(C)')
plt.title('Validation accuracy as a function of C and gamma')
plt.xticks(np.round(np.log10(gamma_range), 2))
plt.yticks(np.round(np.log10(C_range), 2))
plt.show()

plt.show()



