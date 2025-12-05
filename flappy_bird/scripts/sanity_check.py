"""
Quick test script to verify DQN and PPO agents work correctly.
"""

import gymnasium as gym
import numpy as np
import torch
from flappy_bird.agents.dqn import DQNAgent
from flappy_bird.agents.ppo import PPOAgent


def test_dqn():
    """Test DQN agent."""
    print("\n" + "=" * 50)
    print("Testing DQN Agent")
    print("=" * 50)
    
    # Create environment
    env = gym.make("FlappyBirdEnvWithContinuousObs")
    
    # Create agent
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # Run a few episodes
    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        for step in range(100):
            print(obs)
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            agent.store_transition(obs, action, reward, next_obs, terminated)
            agent.train_step()
            
            episode_reward += reward
            obs = next_obs
            
            if terminated or truncated:
                break
        
        agent.decay_epsilon()
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step + 1}")
    
    env.close()
    print("✓ DQN test passed!")


def test_ppo():
    """Test PPO agent."""
    print("\n" + "=" * 50)
    print("Testing PPO Agent")
    print("=" * 50)
    
    # Create environment
    env = gym.make("FlappyBirdEnvWithContinuousObs")
    
    # Create agent
    agent = PPOAgent(state_dim=4, action_dim=2)
    
    # Run a few episodes
    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        while True:
            action, log_prob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            agent.store_transition(obs, action, reward, next_obs, terminated)
            
            episode_reward += reward
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Train on collected trajectory
        advantages, returns = agent.compute_gae_advantages(terminated=terminated)
        agent.train_step(advantages, returns)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    print("✓ PPO test passed!")


if __name__ == "__main__":
    print("Testing implementation of DQN and PPO agents...")
    test_dqn()
    test_ppo()
    print("\n" + "=" * 50)
    print("All tests passed! Ready for comparison experiment.")
    print("=" * 50)
