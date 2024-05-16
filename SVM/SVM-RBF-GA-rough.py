import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

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
ngen=15
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



