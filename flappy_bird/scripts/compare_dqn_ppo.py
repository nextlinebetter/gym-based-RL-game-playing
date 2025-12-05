"""
Compare DQN (Off-policy) vs PPO (On-policy) algorithms on FlappyBird.

实验设计思路：
=============
1. **公平性设置**：
   - 两个算法使用相同的网络容量（128-128维隐藏层）
   - 相同的环境和观测空间
   - 相同的总交互步数上限（100万步）
   - 相同的随机种子进行多次运行（5个不同的种子）

2. **Off-policy (DQN) 特点**：
   - 使用经验回放 (Replay Buffer) 复用历史数据
   - 定期更新目标网络以提高稳定性
   - 探索率ε逐步衰减，后期主要利用学到的策略
   - 样本效率高（同样数据可训练多次）

3. **On-policy (PPO) 特点**：
   - 收集一段轨迹后立即更新（不复用旧数据）
   - 使用GAE计算优势函数，减低方差
   - PPO裁剪目标防止过度更新，提高稳定性
   - 样本效率相对较低但学习稳定

4. **评估指标**：
   - 平均奖励：衡量学习成果
   - 标准差：衡量学习稳定性
   - 收敛速度：多少步达到目标性能
   - 最高性能：训练结束时的最佳成绩

5. **对比结论预期**：
   - DQN：前期学习快，中期波动可能较大，后期稳定
   - PPO：学习平稳，早期可能较DQN慢，后期性能稳定
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import torch
import argparse
from typing import Dict, Any

# Import agents
from flappy_bird.agents.dqn import DQNAgent
from flappy_bird.agents.ppo import PPOAgent

# ===== 超参数配置 =====
# 在这里快速调整两个算法的超参数
HYPERPARAMS = {
    'num_seeds': 1,
    'total_steps': 5_0000,  # 10_0000
    'eval_interval': 5000,
    'eval_episodes': 10,
    
    # DQN 超参数
    'dqn': {
        'state_dim': 4,
        'action_dim': 2,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 0.1,   # 1.0
        'epsilon_end': 0.001,   # 0.01
        'epsilon_decay': 0.995,
        'buffer_size': 10000,
        'batch_size': 64,
        'target_update_freq': 1000,
    },
    
    # PPO 超参数
    'ppo': {
        'state_dim': 4,
        'action_dim': 2,
        'learning_rate': 1e-4,  # 3e-4
        'gamma': 0.99,
        'gae_lambda': 0.9,      # 0.95
        'clip_ratio': 0.1,      # 0.2
        'entropy_coef': 0.005,  # 0.01
        'value_coef': 0.5,
        'n_epochs': 4,          # 10
        'batch_size': 32,       # 64
    }
}


def load_hyperparams(config_path: str = None) -> Dict[str, Any]:
    """
    从JSON文件加载超参数配置。
    
    Args:
        config_path: JSON配置文件路径，如果为None则使用默认配置
    
    Returns:
        超参数字典
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return HYPERPARAMS


def save_hyperparams(config_path: str = "./hyperparams_config.json"):
    """
    保存当前超参数配置到JSON文件。
    
    Args:
        config_path: 保存路径
    """
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(HYPERPARAMS, f, indent=2)
    print(f"✓ 超参数配置已保存到: {config_path}")


def print_hyperparams(hyperparams: Dict[str, Any]):
    """打印当前超参数配置"""
    print("\n" + "=" * 70)
    print("当前超参数配置")
    print("=" * 70)
    print(f"\n通用配置:")
    print(f"  - 种子数 (num_seeds): {hyperparams['num_seeds']}")
    print(f"  - 总步数 (total_steps): {hyperparams['total_steps']:,}")
    print(f"  - 评估间隔 (eval_interval): {hyperparams['eval_interval']:,}")
    print(f"  - 每次评估的环节数 (eval_episodes): {hyperparams['eval_episodes']}")
    
    print(f"\nDQN 超参数:")
    for key, value in hyperparams['dqn'].items():
        print(f"  - {key}: {value}")
    
    print(f"\nPPO 超参数:")
    for key, value in hyperparams['ppo'].items():
        print(f"  - {key}: {value}")
    print()


def run_episode(env, agent, algorithm: str = "dqn", training: bool = True):
    """
    Run one episode with either DQN or PPO agent.
    
    Args:
        env: Environment
        agent: Agent instance (DQN or PPO)
        algorithm: 'dqn' or 'ppo'
        training: Whether in training mode (affects exploration)
    
    Returns:
        episode_reward: Total reward for the episode
        episode_length: Number of steps in the episode
    """
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    if algorithm == "dqn":
        while True:
            # DQN: get action from epsilon-greedy policy
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Store transition in replay buffer
            agent.store_transition(obs, action, reward, next_obs, terminated)
            
            # Train on batch from replay buffer
            if training:
                agent.train_step()
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Decay exploration at end of episode
        if training:
            agent.decay_epsilon()
    
    elif algorithm == "ppo":
        # PPO: collect trajectory
        while True:
            action, log_prob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Train on collected trajectory
        if training:
            advantages, returns = agent.compute_gae_advantages(terminated=terminated)
            agent.train_step(advantages, returns)
    
    return episode_reward, episode_length


def train_agent(env, agent, algorithm: str, total_steps: int = 1000000, 
                eval_interval: int = 5000, eval_episodes: int = 10):
    """
    Train agent until reaching total_steps or max episodes.
    
    Args:
        env: Environment
        agent: Agent instance
        algorithm: 'dqn' or 'ppo'
        total_steps: Maximum training steps
        eval_interval: Evaluate every N steps
        eval_episodes: Number of episodes per evaluation
    
    Returns:
        training_results: Dict with training statistics
    """
    training_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_rewards': [],
        'eval_steps': [],
        'best_reward': -np.inf,
    }
    
    total_steps_count = 0
    episode_count = 0
    
    with tqdm(total=total_steps, desc=f"Training {algorithm.upper()}") as pbar:
        while total_steps_count < total_steps:
            # Training episode
            episode_reward, episode_length = run_episode(env, agent, algorithm, training=True)
            
            agent.episode_rewards.append(episode_reward)
            training_results['episode_rewards'].append(episode_reward)
            training_results['episode_lengths'].append(episode_length)
            
            total_steps_count += episode_length
            episode_count += 1
            
            pbar.update(episode_length)
            
            # Periodic evaluation
            if episode_count % max(1, eval_interval // 100) == 0:  # Evaluate more frequently
                eval_rewards = []
                for _ in range(eval_episodes):
                    reward, _ = run_episode(env, agent, algorithm, training=False)
                    eval_rewards.append(reward)
                
                avg_eval_reward = np.mean(eval_rewards)
                training_results['eval_rewards'].append(avg_eval_reward)
                training_results['eval_steps'].append(total_steps_count)
                
                if avg_eval_reward > training_results['best_reward']:
                    training_results['best_reward'] = avg_eval_reward
                
                pbar.set_postfix({
                    'episode': episode_count,
                    'avg_reward': f"{np.mean(agent.episode_rewards[-10:]):.2f}",
                    'eval_reward': f"{avg_eval_reward:.2f}"
                })
    
    print(f"\n{algorithm.upper()} Training Complete!")
    print(f"  Episodes: {episode_count}")
    print(f"  Total Steps: {total_steps_count}")
    print(f"  Best Eval Reward: {training_results['best_reward']:.2f}")
    print(f"  Final Avg Reward (last 10): {np.mean(agent.episode_rewards[-10:]):.2f}")
    
    return training_results


def compare_algorithms(hyperparams: Dict[str, Any] = None):
    """
    Compare DQN vs PPO across multiple random seeds.
    
    Args:
        hyperparams: 超参数配置字典，如果为None则使用全局HYPERPARAMS
    """
    if hyperparams is None:
        hyperparams = HYPERPARAMS
    
    num_seeds = hyperparams['num_seeds']
    total_steps = hyperparams['total_steps']
    eval_interval = hyperparams['eval_interval']
    eval_episodes = hyperparams['eval_episodes']
    
    print("=" * 70)
    print("DQN (Off-policy) vs PPO (On-policy) Comparison on FlappyBird")
    print("=" * 70)
    
    print_hyperparams(hyperparams)
    
    # Storage for results
    dqn_all_results = []
    ppo_all_results = []
    
    for seed in range(num_seeds):
        print(f"\n[Seed {seed + 1}/{num_seeds}]")
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environment
        env = gym.make("FlappyBirdEnvWithContinuousObs")
        
        # ===== DQN Training =====
        print("\n--- Training DQN ---")
        dqn_agent = DQNAgent(**hyperparams['dqn'])
        
        dqn_results = train_agent(env, dqn_agent, "dqn", 
                                  total_steps=total_steps,
                                  eval_interval=eval_interval,
                                  eval_episodes=eval_episodes)
        dqn_all_results.append(dqn_results)
        
        # Reset environment
        env.close()
        env = gym.make("FlappyBirdEnvWithContinuousObs")
        
        # ===== PPO Training =====
        print("\n--- Training PPO ---")
        ppo_agent = PPOAgent(**hyperparams['ppo'])
        
        ppo_results = train_agent(env, ppo_agent, "ppo", 
                                  total_steps=total_steps,
                                  eval_interval=eval_interval,
                                  eval_episodes=eval_episodes)
        ppo_all_results.append(ppo_results)
        
        env.close()
    
    # ===== Aggregate Results =====
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    # Extract evaluation rewards and steps
    dqn_eval_rewards_all = [r['eval_rewards'] for r in dqn_all_results]
    ppo_eval_rewards_all = [r['eval_rewards'] for r in ppo_all_results]
    dqn_eval_steps_all = [r['eval_steps'] for r in dqn_all_results]
    ppo_eval_steps_all = [r['eval_steps'] for r in ppo_all_results]
    
    # Find maximum evaluation points for each algorithm
    max_dqn_len = max(len(r) for r in dqn_eval_rewards_all)
    max_ppo_len = max(len(r) for r in ppo_eval_rewards_all)
    
    # Determine the reference eval_steps (use the longest one)
    if max_dqn_len >= max_ppo_len:
        # Find the eval_steps with maximum length from DQN
        dqn_eval_steps = dqn_eval_steps_all[max(range(len(dqn_eval_steps_all)), 
                                                   key=lambda i: len(dqn_eval_steps_all[i]))]
        ref_eval_steps = dqn_eval_steps
        ref_len = max_dqn_len
    else:
        # Find the eval_steps with maximum length from PPO
        ppo_eval_steps = ppo_eval_steps_all[max(range(len(ppo_eval_steps_all)), 
                                                   key=lambda i: len(ppo_eval_steps_all[i]))]
        ref_eval_steps = ppo_eval_steps
        ref_len = max_ppo_len
    
    # Pad all eval_rewards to the reference length
    dqn_eval_rewards_all = [r + [r[-1]] * (ref_len - len(r)) if len(r) > 0 else [0] * ref_len 
                            for r in dqn_eval_rewards_all]
    ppo_eval_rewards_all = [r + [r[-1]] * (ref_len - len(r)) if len(r) > 0 else [0] * ref_len 
                            for r in ppo_eval_rewards_all]
    
    # Use reference eval_steps
    eval_steps = ref_eval_steps
    
    dqn_eval_rewards = np.array(dqn_eval_rewards_all)
    ppo_eval_rewards = np.array(ppo_eval_rewards_all)
    
    dqn_mean = dqn_eval_rewards.mean(axis=0)
    dqn_std = dqn_eval_rewards.std(axis=0)
    ppo_mean = ppo_eval_rewards.mean(axis=0)
    ppo_std = ppo_eval_rewards.std(axis=0)
    
    # Print statistics
    print(f"\nDQN Results ({num_seeds} runs):")
    print(f"  Best Avg Reward: {dqn_mean[-1]:.2f} ± {dqn_std[-1]:.2f}")
    print(f"  Overall Best:    {dqn_mean.max():.2f}")
    
    print(f"\nPPO Results ({num_seeds} runs):")
    print(f"  Best Avg Reward: {ppo_mean[-1]:.2f} ± {ppo_std[-1]:.2f}")
    print(f"  Overall Best:    {ppo_mean.max():.2f}")
    
    winner = "DQN" if dqn_mean[-1] > ppo_mean[-1] else "PPO"
    print(f"\nWinner: {winner}")
    
    # Save results
    os.makedirs("./results/flappy_bird/comparison", exist_ok=True)
    results_summary = {
        'hyperparams': hyperparams,
        'dqn_final_reward': float(dqn_mean[-1]),
        'dqn_final_std': float(dqn_std[-1]),
        'dqn_best_reward': float(dqn_mean.max()),
        'ppo_final_reward': float(ppo_mean[-1]),
        'ppo_final_std': float(ppo_std[-1]),
        'ppo_best_reward': float(ppo_mean.max()),
        'winner': winner,
    }
    
    with open("./results/flappy_bird/comparison/results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Plot results
    eval_steps = np.array(eval_steps)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(eval_steps, dqn_mean, label='DQN', color='blue', linewidth=2)
    plt.fill_between(eval_steps, dqn_mean - dqn_std, dqn_mean + dqn_std, 
                      alpha=0.3, color='blue')
    
    plt.plot(eval_steps, ppo_mean, label='PPO', color='red', linewidth=2)
    plt.fill_between(eval_steps, ppo_mean - ppo_std, ppo_mean + ppo_std, 
                      alpha=0.3, color='red')
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Evaluation Reward (Mean ± Std)', fontsize=12)
    plt.title('DQN vs PPO on FlappyBird ({} Random Seeds)'.format(num_seeds), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/flappy_bird/comparison/dqn_vs_ppo.png", dpi=150)
    print(f"\nPlot saved to ./results/flappy_bird/comparison/dqn_vs_ppo.png")
    
    return results_summary


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(
        description='DQN vs PPO 比较实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            使用示例:
            python compare_dqn_ppo.py                           # 使用默认超参数
            python compare_dqn_ppo.py --config config.json      # 加载配置文件
            python compare_dqn_ppo.py --save-config             # 保存当前超参数配置
            python compare_dqn_ppo.py --seed 5 --steps 500000   # 调整特定参数
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='加载JSON格式的超参数配置文件')
    parser.add_argument('--save-config', action='store_true',
                        help='保存当前超参数配置到 hyperparams_config.json')
    parser.add_argument('--seed', type=int, default=None,
                        help='覆盖配置中的种子数')
    parser.add_argument('--steps', type=int, default=None,
                        help='覆盖配置中的总步数')
    parser.add_argument('--dqn-lr', type=float, default=None,
                        help='DQN学习率')
    parser.add_argument('--ppo-lr', type=float, default=None,
                        help='PPO学习率')
    
    args = parser.parse_args()
    
    # 加载超参数
    hyperparams = load_hyperparams(args.config)
    
    # 命令行参数覆盖配置文件
    if args.seed is not None:
        hyperparams['num_seeds'] = args.seed
    if args.steps is not None:
        hyperparams['total_steps'] = args.steps
    if args.dqn_lr is not None:
        hyperparams['dqn']['learning_rate'] = args.dqn_lr
    if args.ppo_lr is not None:
        hyperparams['ppo']['learning_rate'] = args.ppo_lr
    
    # 保存配置
    if args.save_config:
        save_hyperparams()
        return
    
    # 运行比较
    results = compare_algorithms(hyperparams)


if __name__ == "__main__":
    main()
