import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import pickle
# 导入您自己实现的 Q-Learning Agent 类
from flappy_bird.agents.qlearning import FlappyBirdAgent 
import flappy_bird # 注册环境 (确保 flappy_bird 包在 Python 路径中)

# 设置 Matplotlib 后端，防止无头服务器报错
plt.switch_backend('Agg')

# ⚠️ 必须从您的 Q-Learning Agent 类中导入用于离散化的参数 ⚠️
# 这两个参数必须与 FlappyBirdAgent 类中定义的值一致，否则绘图不准确
Y_DIFF_BIN_SIZE = 50  
VELOCITY_BIN_SIZE = 1


def plot_trajectory(agent, env_name="FlappyBirdEnvWithContinuousObs"):
    """
    1. 飞行轨迹图：画出小鸟的飞行高度 vs 管子缝隙高度。
    """
    print("Generating Trajectory Plot...")
    # 注意：Q-Learning agent 需要与训练时相同的环境
    env = gym.make(env_name) 
    obs, _ = env.reset()
    
    bird_y_history = []
    gap_y_history = []
    actions = []
    
    done = False
    step = 0
    # 跑一局，限制最多 500 步，防止跑太久
    # Flappy Bird 步数较多，这里限制为 10000 步
    while not done and step < 10000:
        # 获取原始位置信息 (需要访问环境私有变量)
        bird_real_y = env.unwrapped._bird.y
        if len(env.unwrapped._pipes) > 0:
            # 获取当前水管的缝隙中心Y坐标
            gap_center = env.unwrapped._pipes[0].height + env.unwrapped._pipes[0].gap / 2
        else:
            gap_center = 256 # 默认中心
            
        bird_y_history.append(bird_real_y)
        gap_y_history.append(gap_center)
        
        # Q-Learning Agent 获取动作
        # 注意：这里我们使用 get_action，但将 epsilon 设置为 0，以确保纯粹的利用（exploitation）
        # 我们需要在 FlappyBirdAgent 中添加一个参数来控制探索率，或者临时设置 agent.epsilon = 0
        original_epsilon = agent.epsilon
        agent.epsilon = 0 # 强制利用策略进行评估
        action = agent.get_action(obs)
        agent.epsilon = original_epsilon # 恢复 epsilon
        
        actions.append(action)
        
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    
    # 开始绘图
    plt.figure(figsize=(12, 6))
    
    # 1. 画管子中心线
    plt.plot(gap_y_history, color='green', linestyle='--', linewidth=2, label='Pipe Gap Center')
    
    # 2. 画小鸟轨迹
    plt.plot(bird_y_history, color='orange', linewidth=2, label='Bird Height')
    
    # 3. 标记跳跃点
    jump_steps = [i for i, a in enumerate(actions) if a == 1]
    jump_y = [bird_y_history[i] for i in jump_steps]
    plt.scatter(jump_steps, jump_y, color='red', s=10, marker='^', label='Jump Action', zorder=5)

    plt.title(f"Q-Agent Flight Trajectory (Score: {step})")
    plt.xlabel("Time Steps")
    plt.ylabel("Height (Pixels)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y轴反转，因为游戏坐标系 0 在上面
    plt.gca().invert_yaxis()
    
    save_path = "./results/qlearning_trajectory_plot.png"
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


def plot_policy_heatmap(agent):
    """
    2. 策略热力图：展示 Q-Table 在离散状态下的决策。
    X轴: 离散化的速度
    Y轴: 离散化的垂直高度差
    颜色: 最佳动作 (跳跃 vs 不跳跃)
    """
    print("Generating Q-Table Decision Map...")
    
    # 离散化后的状态范围（假设我们只需要显示这些范围）
    y_min, y_max = -10 * Y_DIFF_BIN_SIZE, 10 * Y_DIFF_BIN_SIZE
    vel_min, vel_max = -10 * VELOCITY_BIN_SIZE, 10 * VELOCITY_BIN_SIZE
    
    resolution = 100 # 绘图网格分辨率
    
    # 连续网格点
    y_diffs_cont = np.linspace(y_min, y_max, resolution)
    vels_cont = np.linspace(vel_min, vel_max, resolution)
    
    # 决策网格：存储最佳动作 (1: 跳跃, 0: 不跳跃, -1: 未访问)
    decision_grid = np.full((resolution, resolution), -1.0)
    
    # 遍历网格
    for i, y_cont in enumerate(y_diffs_cont):
        for j, vel_cont in enumerate(vels_cont):
            
            # 1. 对连续值进行离散化 (模拟 agent._obs_to_key 的逻辑)
            y_discrete = int(np.clip(y_cont // Y_DIFF_BIN_SIZE, -10, 10))
            vel_discrete = int(np.clip(vel_cont // VELOCITY_BIN_SIZE, -10, 10))
            
            state_key = (y_discrete, vel_discrete)
            
            # 2. 从 Q-Table 中获取 Q 值
            # 注意：如果状态未被访问，defaultdict 会返回 np.zeros，argmax 会是 0 (不跳)
            q_values = agent.q_values[state_key]
            
            # 3. 确定最佳动作 (Exploitation)
            best_action = np.argmax(q_values)
            
            # 4. 存储到决策网格
            # 如果两个 Q 值都是 0 (未访问)，则 best_action 为 0，但我们用 -1 标记未访问状态会更清晰
            if np.all(q_values == 0):
                 decision_grid[i, j] = 0.5 # 未访问状态，用中间颜色表示
            else:
                 decision_grid[i, j] = best_action

    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 使用 imshow 绘制热力图 (这里使用 cmap=bwr 来区分 0 和 1)
    # 我们将 y_diffs_cont 和 vels_cont 作为 extent
    img = plt.imshow(
        decision_grid, 
        extent=[vel_min, vel_max, y_min, y_max], # [x_min, x_max, y_min, y_max]
        origin='lower', 
        aspect='auto', 
        cmap='bwr', # Blue-White-Red (蓝色=不跳, 红色=跳)
        vmin=0, vmax=1
    )
    
    cbar = plt.colorbar(img, ticks=[0, 1], label='Optimal Action')
    cbar.ax.set_yticklabels(['0 (Stay)', '1 (Jump)'])
    
    plt.title("Q-Table Decision Boundary (Optimal Policy)")
    plt.xlabel(f"Vertical Velocity (Binned by {VELOCITY_BIN_SIZE})")
    plt.ylabel(f"Height Diff to Pipe Center (Binned by {Y_DIFF_BIN_SIZE})")
    
    # 画一条中心线
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.text(vel_min + 5, -50, "Bird is too HIGH", color='black', fontsize=10)
    plt.text(vel_min + 5, 50, "Bird is too LOW", color='black', fontsize=10)
    
    save_path = "./results/qlearning_policy_heatmap.png"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    
    # 1. 自动适配环境
    env_name = "FlappyBirdEnvWithContinuousObs"
    env = gym.make(env_name)
    
    # 2. 加载 Q-Learning 模型
    model_path = "./results/flappy-bird/qlearning/best_agent.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Q-Learning Model not found at {model_path}")
        print("Please train your Q-Learning agent first.")
        exit()
        
    # Q-Learning Agent 使用 load 方法 (需要 pickle)
    agent = FlappyBirdAgent.load(model_path, env)
    
    # 3. 生成图表
    os.makedirs("./results", exist_ok=True)
    plot_trajectory(agent, env_name)
    plot_policy_heatmap(agent)
