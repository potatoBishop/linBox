import numpy as np

# pythonCopy codeimport numpy as np
# 粒子群优化算法类
class PSO:
    def __init__(self, num_particles, num_dimensions, max_iter, target_func):
        self.num_particles = num_particles  # 粒子数量
        self.num_dimensions = num_dimensions  # 解的维度
        self.max_iter = max_iter  # 最大迭代次数
        self.target_func = target_func  # 目标函数
        self.particles = np.random.uniform(-5, 5, size=(num_particles, num_dimensions))  # 初始化粒子位置
        self.velocities = np.zeros((num_particles, num_dimensions))  # 初始化粒子速度
        self.best_positions = np.copy(self.particles)  # 初始化粒子的个体最佳位置
        self.global_best_position = None  # 全局最佳位置
    # 粒子群优化算法的迭代过程
    def optimize(self):
        for i in range(self.max_iter):
            # 更新粒子速度和位置
            self.update_velocity()
            self.update_position()
            # 计算适应度值
            fitness = self.target_func(self.particles)
            # 更新个体最佳位置和全局最佳位置
            self.update_best_positions(fitness)
            self.update_global_best_position(fitness)
    # 更新粒子速度
    def update_velocity(self):
        inertia_weight = 0.9  # 惯性权重
        cognitive_weight = 2  # 认知权重
        social_weight = 2  # 社会权重
        for i in range(self.num_particles):
            r1 = np.random.random()  # 随机数1
            r2 = np.random.random()  # 随机数2
            cognitive_component = cognitive_weight * r1 * (self.best_positions[i] - self.particles[i])
            social_component = social_weight * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component
    # 更新粒子位置
    def update_position(self):
        self.particles += self.velocities
    # 更新个体最佳位置
    def update_best_positions(self, fitness):
        mask = fitness < self.target_func(self.best_positions)
        self.best_positions[mask] = self.particles[mask]
    # 更新全局最佳位置
    def update_global_best_position(self, fitness):
        best_index = np.argmin(fitness)
        self.global_best_position = self.particles[best_index]
# 示例目标函数：Rastrigin函数
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=1)
# 运行粒子群优化算法
pso = PSO(num_particles=50, num_dimensions=2, max_iter=100, target_func=rastrigin)
pso.optimize()
# 打印最优解和最优适应度值
print("最优解：", pso.global_best_position)
print("最优适应度值：", rastrigin(pso.global_best_position))