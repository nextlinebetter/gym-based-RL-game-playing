# DQN vs PPO 对比实验设计文档

## 1. 实验目标

在 FlappyBird 游戏环境上对比**DQN (Deep Q-Network)** 和 **PPO (Proximal Policy Optimization)** 两种强化学习算法的性能，探讨**Off-policy**和**On-policy**算法的优缺点。

## 2. 算法对比

### DQN (Off-policy)

- **学习特点**：

  - 从任何历史行为数据中学习（历史数据可重复利用）
  - 使用经验回放（Replay Buffer）打破样本相关性
  - 目标网络稳定 Q 值目标
  - 探索率 ε 逐步衰减

- **优点**：

  1. **高样本效率**：同一条轨迹可训练多次
  2. **稳定性好**：目标网络和经验回放减少波动
  3. **易于并行化**：回放缓冲区支持异步采样

- **缺点**：
  1. 超参数敏感（学习率、ε 衰减、缓冲区大小）
  2. 大规模探索可能低效（ε-greedy 较粗糙）
  3. Q 值高估问题（Double DQN 可改进，但此处未实现）

### PPO (On-policy)

- **学习特点**：

  - 仅从当前策略的轨迹中学习（不复用旧数据）
  - 使用 GAE (Generalized Advantage Estimation) 计算优势函数
  - PPO 裁剪约束防止过度更新
  - 每轨迹多个 epoch 训练

- **优点**：

  1. **学习稳定**：PPO 裁剪防止剧烈策略波动
  2. **易于调参**：对超参数不如 DQN 敏感
  3. **高效探索**：策略熵正则化鼓励探索
  4. **收敛性好**：GAE 平衡偏差-方差

- **缺点**：
  1. **样本效率较低**：数据一次性使用
  2. 需要多 step 计算回报（内存占用）
  3. 计算梯度多次（N_epochs × batch_size）

## 3. 实验设置

### 环境配置

- **环境**：FlappyBirdEnvWithContinuousObs
- **观测空间**：4 维连续状态 [bird_y, bird_v, pipe_x, pipe_y_diff]
- **动作空间**：2 维离散 [0=stay, 1=flap]
- **奖励设计**：
  - 通过管道：+1.0
  - 存活每步：+0.1
  - 碰撞：-1.0

### 网络架构（保持一致）

```
输入 (4维状态)
  ↓
FC1 (4 → 128, ReLU)
  ↓
FC2 (128 → 128, ReLU)
  ↓
输出 (128 → action_dim或value)
```

### 超参数设置

#### DQN 参数

| 参数          | 值          | 说明             |
| ------------- | ----------- | ---------------- |
| Learning Rate | 1e-3        | 优化器学习率     |
| Gamma (γ)     | 0.99        | 折扣因子         |
| Epsilon Start | 1.0         | 初始探索率       |
| Epsilon End   | 0.01        | 最小探索率       |
| Epsilon Decay | 0.995       | 每步衰减因子     |
| Buffer Size   | 10,000      | 回放缓冲区容量   |
| Batch Size    | 64          | 训练批量大小     |
| Target Update | 1,000 steps | 更新目标网络频率 |

#### PPO 参数

| 参数          | 值   | 说明                      |
| ------------- | ---- | ------------------------- |
| Learning Rate | 3e-4 | 优化器学习率              |
| Gamma (γ)     | 0.99 | 折扣因子                  |
| GAE Lambda    | 0.95 | GAE 参数（偏差-方差权衡） |
| Clip Ratio    | 0.2  | PPO 裁剪范围 ±20%         |
| Entropy Coef  | 0.01 | 熵正则化系数              |
| Value Coef    | 0.5  | 价值函数损失系数          |
| N Epochs      | 10   | 每轨迹训练轮数            |
| Batch Size    | 64   | 小批量大小                |

### 训练条件

- **总训练步数**：500,000 步
- **评估间隔**：每 100 个 episode 评估一次
- **评估方式**：运行 10 个 episode（无探索）计算平均奖励
- **随机种子**：5 个不同的种子（保证结果可复现且有统计意义）

## 4. 评估指标

### 主要指标

1. **最终性能** (Final Reward)

   - 定义：最后 10 个 episode 的平均奖励
   - 衡量：算法的最终收敛水平

2. **最佳性能** (Peak Reward)

   - 定义：整个训练过程中的最高平均奖励
   - 衡量：算法能达到的上限

3. **稳定性** (Stability)

   - 定义：多次运行的标准差
   - 衡量：结果的可复现性和鲁棒性

4. **收敛速度** (Convergence Speed)
   - 定义：达到特定奖励阈值的步数
   - 衡量：样本效率和学习速度

### 二级指标

- 训练中平均奖励的平滑度
- 评估时的奖励波动幅度
- 总训练时间

## 5. 实验流程

```
对于每个随机种子 (seed = 0 to 4):
  1. 设置随机种子 (numpy, torch)
  2. 创建新的环境实例

  3. 训练DQN:
     - 初始化DQN Agent
     - 循环直到500,000步:
       a. 执行一个episode
       b. 存储转移到回放缓冲区
       c. 从缓冲区采样训练一个batch
       d. 衰减epsilon
       e. 定期评估并记录

  4. 重置环境

  5. 训练PPO:
     - 初始化PPO Agent
     - 循环直到500,000步:
       a. 收集一个完整episode
       b. 计算GAE优势函数
       c. 用PPO目标训练N个epoch
       d. 定期评估并记录

结束后:
  1. 聚合所有种子的结果
  2. 计算均值、标准差
  3. 绘制对比曲线
  4. 输出统计分析结果
```

## 6. 预期结论

### DQN 可能表现

- **前期**：快速学习，因为能复用数据
- **中期**：可能出现性能波动（Q 值估计误差）
- **后期**：稳定收敛，最终性能可能较高

### PPO 可能表现

- **前期**：学习较平稳，性能提升可能较慢
- **中期**：学习稳定，性能均匀提升
- **后期**：收敛平稳，最终性能稳定

### 关键对比点

1. **样本效率**：DQN 应该更高（同样数据多次训练）
2. **学习稳定性**：PPO 应该更好（策略更新更温和）
3. **最终性能**：可能接近，取决于超参数调优程度
4. **收敛速度**：DQN 可能更快达到较高性能

## 7. 运行说明

### 快速测试

```bash
cd gym-based-RL-game-playing
python -m flappy_bird.scripts.sanity_check
```

### 运行完整对比实验

```bash
python -m flappy_bird.scripts.compare_dqn_ppo
```

### 输出结果

- `results/flappy_bird/comparison/dqn_vs_ppo.png`：对比图
- `results/flappy_bird/comparison/results_summary.json`：统计数据

## 8. 理论基础

### DQN 的 Bellman 方程

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

- 用神经网络逼近 Q 函数
- 目标网络稳定：$Q_{target} = r + \gamma \max Q_{target}(s', a')$

### PPO 的目标函数

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$

其中：

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 是重要性采样比
- $\hat{A}_t$ 是通过 GAE 估计的优势函数
- 裁剪防止过大的策略更新

## 9. 局限性和扩展

### 当前局限

1. 未实现 Double DQN（防止 Q 值高估）
2. 未实现 Dueling DQN（分离价值和优势）
3. PPO 未使用独立的价值网络
4. 网络架构简单（可尝试 CNN 处理图像观测）

### 可能的扩展

1. 比较不同网络容量的影响
2. 使用图像观测而不是特征观测
3. 实现优先经验回放 (PER)
4. 添加更复杂的探索策略
