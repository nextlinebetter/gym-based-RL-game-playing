import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple
import os


class DQNNetwork(nn.Module):
    """Deep Q-Network for continuous state observations."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    """DQN (Deep Q-Network) Agent - Off-policy algorithm."""
    
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = None,
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space (4 for FlappyBird: y, v, pipe_x, pipe_y_diff)
            action_dim: Dimension of action space (2: stay or flap)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate per step
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            target_update_freq: Update target network every N steps
            device: 'cuda' or 'cpu'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training stats
        self.train_steps = 0
        self.episode_rewards = []
        self.training_losses = []
        
    def _obs_to_tensor(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Convert observation dict to tensor."""
        state = np.concatenate([
            obs["bird_y"],
            obs["bird_v"],
            obs["pipe_x"],
            obs["pipe_y_diff"]
        ])
        return torch.FloatTensor(state).to(self.device)
    
    def get_action(self, obs: Dict[str, Any]) -> int:
        """
        Select action using epsilon-greedy strategy.
        Off-policy: explore during training.
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Exploit
        with torch.no_grad():
            state = self._obs_to_tensor(obs)
            q_values = self.q_network(state)
            return q_values.argmax(dim=0).item()
    
    def store_transition(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        next_obs: Dict[str, Any],
        terminated: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.append((obs, action, reward, next_obs, terminated))
    
    def train_step(self):
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # Unpack batch
        obss, actions, rewards, next_obss, terminateds = zip(*batch)
        
        states = torch.stack([self._obs_to_tensor(obs) for obs in obss])
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack([self._obs_to_tensor(obs) for obs in next_obss])
        terminateds = torch.BoolTensor(terminateds).to(self.device)
        
        # Compute Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~terminateds)
        
        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_losses.append(loss.item())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str = "./results/flappy_bird/dqn/agent.pt"):
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses,
        }, filepath)
        print(f"DQN Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, state_dim: int = 4, action_dim: int = 2):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        agent = cls(state_dim=state_dim, action_dim=action_dim)
        agent.q_network.load_state_dict(checkpoint['q_network'])
        agent.target_network.load_state_dict(checkpoint['target_network'])
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
        agent.epsilon = checkpoint['epsilon']
        agent.train_steps = checkpoint['train_steps']
        agent.episode_rewards = checkpoint.get('episode_rewards', [])
        agent.training_losses = checkpoint.get('training_losses', [])
        print(f"DQN Agent loaded from {filepath}")
        return agent
