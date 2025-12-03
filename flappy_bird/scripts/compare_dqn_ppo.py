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

# Import agents
from flappy_bird.agents.dqn import DQNAgent
from flappy_bird.agents.ppo import PPOAgent

NUM_SEEDS = 1
TOTAL_STEPS = 10_0000  # 50_0000


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


def compare_algorithms(num_seeds: int = 5, total_steps: int = 500000):
    """
    Compare DQN vs PPO across multiple random seeds.
    
    Args:
        num_seeds: Number of random seed runs
        total_steps: Total training steps per seed
    """
    print("=" * 70)
    print("DQN (Off-policy) vs PPO (On-policy) Comparison on FlappyBird")
    print("=" * 70)
    
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
        dqn_agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=1000,
        )
        
        dqn_results = train_agent(env, dqn_agent, "dqn", total_steps=total_steps)
        dqn_all_results.append(dqn_results)
        
        # Reset environment
        env.close()
        env = gym.make("FlappyBirdEnvWithContinuousObs")
        
        # ===== PPO Training =====
        print("\n--- Training PPO ---")
        ppo_agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            n_epochs=10,
            batch_size=64,
        )
        
        ppo_results = train_agent(env, ppo_agent, "ppo", total_steps=total_steps)
        ppo_all_results.append(ppo_results)
        
        env.close()
    
    # ===== Aggregate Results =====
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    # Extract evaluation rewards
    dqn_eval_rewards_all = [r['eval_rewards'] for r in dqn_all_results]
    ppo_eval_rewards_all = [r['eval_rewards'] for r in ppo_all_results]
    
    # Pad to same length
    max_len = max(len(dqn_eval_rewards_all[0]), len(ppo_eval_rewards_all[0]))
    dqn_eval_rewards_all = [r + [r[-1]] * (max_len - len(r)) for r in dqn_eval_rewards_all]
    ppo_eval_rewards_all = [r + [r[-1]] * (max_len - len(r)) for r in ppo_eval_rewards_all]
    
    dqn_eval_rewards = np.array(dqn_eval_rewards_all)
    ppo_eval_rewards = np.array(ppo_eval_rewards_all)
    
    dqn_mean = dqn_eval_rewards.mean(axis=0)
    dqn_std = dqn_eval_rewards.std(axis=0)
    ppo_mean = ppo_eval_rewards.mean(axis=0)
    ppo_std = ppo_eval_rewards.std(axis=0)
    
    # Print statistics
    print(f"\nDQN Results (5 runs):")
    print(f"  Best Avg Reward: {dqn_mean[-1]:.2f} ± {dqn_std[-1]:.2f}")
    print(f"  Overall Best:    {dqn_mean.max():.2f}")
    
    print(f"\nPPO Results (5 runs):")
    print(f"  Best Avg Reward: {ppo_mean[-1]:.2f} ± {ppo_std[-1]:.2f}")
    print(f"  Overall Best:    {ppo_mean.max():.2f}")
    
    winner = "DQN" if dqn_mean[-1] > ppo_mean[-1] else "PPO"
    print(f"\nWinner: {winner}")
    
    # Save results
    os.makedirs("./results/flappy_bird/comparison", exist_ok=True)
    results_summary = {
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
    eval_steps = dqn_all_results[0]['eval_steps']
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(eval_steps, dqn_mean, label='DQN', color='blue', linewidth=2)
    plt.fill_between(eval_steps, dqn_mean - dqn_std, dqn_mean + dqn_std, 
                      alpha=0.3, color='blue')
    
    plt.plot(eval_steps, ppo_mean, label='PPO', color='red', linewidth=2)
    plt.fill_between(eval_steps, ppo_mean - ppo_std, ppo_mean + ppo_std, 
                      alpha=0.3, color='red')
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Evaluation Reward (Mean ± Std)', fontsize=12)
    plt.title('DQN vs PPO on FlappyBird (5 Random Seeds)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/flappy_bird/comparison/dqn_vs_ppo.png", dpi=150)
    print(f"\nPlot saved to ./results/flappy_bird/comparison/dqn_vs_ppo.png")
    
    return results_summary


if __name__ == "__main__":
    # Run comparison
    results = compare_algorithms(num_seeds=NUM_SEEDS, total_steps=TOTAL_STEPS)
