import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple
import os


class PPONetwork(nn.Module):
    """Actor-Critic Network for PPO."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPONetwork, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Actor: output logits for action distribution
        action_logits = self.actor(x)
        
        # Critic: output state value
        value = self.critic(x)
        
        return action_logits, value


class PPOAgent:
    """PPO (Proximal Policy Optimization) Agent - On-policy algorithm."""
    
    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 2,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = None,
    ):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (bias-variance tradeoff)
            clip_ratio: PPO clipping ratio for policy
            entropy_coef: Coefficient for entropy regularization
            value_coef: Coefficient for value function loss
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs to train per batch
            batch_size: Batch size for training
            device: 'cuda' or 'cpu'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Network
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminateds = []
        
        # Training stats
        self.episode_rewards = []
        self.training_losses = []
        self.train_steps = 0
        
    def _obs_to_tensor(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Convert observation dict to tensor."""
        state = np.concatenate(list(obs.values()))
        return torch.FloatTensor(state).to(self.device)
    
    def get_action(self, obs: Dict[str, Any]) -> Tuple[int, float]:
        """
        Select action using current policy.
        On-policy: always sample from current policy during training.
        
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
        """
        with torch.no_grad():
            state = self._obs_to_tensor(obs)
            action_logits, value = self.network(state)
            
            # Sample from policy
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def store_transition(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        next_obs: Dict[str, Any],
        terminated: bool,
    ):
        """Store transition in trajectory buffer."""
        state = self._obs_to_tensor(obs)
        
        # Get value estimate
        with torch.no_grad():
            _, value = self.network(state)
        
        # Get action log probability
        action_logits, _ = self.network(state)
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        self.terminateds.append(terminated)
    
    def compute_gae_advantages(self, next_obs: Dict[str, Any] = None, terminated: bool = False):
        """
        Compute GAE (Generalized Advantage Estimation) advantages and returns.
        
        Args:
            next_obs: Observation at end of trajectory (for bootstrap)
            terminated: Whether trajectory ended due to episode termination
        """
        # Bootstrap value
        if next_obs is not None:
            next_state = self._obs_to_tensor(next_obs)
            with torch.no_grad():
                _, next_value = self.network(next_state)
            next_value = next_value.item()
        else:
            next_value = 0.0
        
        if terminated:
            next_value = 0.0
        
        # Compute advantages and returns
        values = self.values + [next_value]
        advantages = []
        gae = 0.0
        
        # Backward pass for GAE
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return torch.tensor(advantages, dtype=torch.float32).to(self.device), \
               torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def train_step(self, advantages: torch.Tensor, returns: torch.Tensor):
        """
        Train on collected trajectory using PPO.
        
        Args:
            advantages: Computed advantages
            returns: Computed returns (advantages + values)
        """
        # Stack trajectories
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            # Mini-batch training
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_normalized[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_logits, values = self.network(batch_states)
                values = values.squeeze()
                
                # Compute new log probs and entropy
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.training_losses.append(loss.item())
        
        self.train_steps += 1
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminateds = []
    
    def save(self, filepath: str = "./results/flappy_bird/ppo/agent.pt"):
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses,
        }, filepath)
        print(f"PPO Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, state_dim: int = 3, action_dim: int = 2):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(state_dim=state_dim, action_dim=action_dim)
        agent.network.load_state_dict(checkpoint['network'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.train_steps = checkpoint['train_steps']
        agent.episode_rewards = checkpoint.get('episode_rewards', [])
        agent.training_losses = checkpoint.get('training_losses', [])
        print(f"PPO Agent loaded from {filepath}")
        return agent
