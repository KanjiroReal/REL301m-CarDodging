import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List
import json
from enum import Enum
import torch.nn.functional as F
import os

class LearningMethod(Enum):
    DQN = "dqn"
    MONTE_CARLO = "monte_carlo" 
    TD = "td"

class DQNNetwork(nn.Module):
    """Mạng neural cho DQN"""
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int, network_config: dict):
        super().__init__()
        
        # Tạo CNN layers từ config
        cnn_layers = []
        current_channels = input_shape[0]
        
        for layer_config in network_config['cnn_layers']:
            cnn_layers.extend([
                nn.Conv2d(current_channels, 
                         layer_config['filters'], 
                         kernel_size=layer_config['kernel_size'], 
                         stride=layer_config['stride']),
                nn.ReLU()
            ])
            current_channels = layer_config['filters']
            
        cnn_layers.append(nn.Flatten())
        self.features = nn.Sequential(*cnn_layers)
        
        # Tính kích thước feature sau CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            n_flatten = self.features(dummy_input).shape[1]
        
        # Tạo FC layers từ config
        fc_layers = []
        current_size = n_flatten
        
        for size in network_config['fc_layers']:
            fc_layers.extend([
                nn.Linear(current_size, size),
                nn.ReLU()
            ])
            current_size = size
            
        # Layer output cuối cùng
        fc_layers.append(nn.Linear(current_size, n_actions))
        
        self.fc = nn.Sequential(*fc_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(x)

class ReplayBuffer:
    """Bộ nhớ để lưu trữ và lấy mẫu các transition"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """Agent DQN"""
    def __init__(self, config_path: str):
        # Đọc config từ file JSON
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.agent_config = config['agent_config']
        self.training_config = config['training_config']
        
        # Khởi tạo các thành phần
        self.learning_method = LearningMethod(self.training_config['learning_method'])
        self.episode_buffer = []
        self.batch_size = self.agent_config['batch_size']
        self.gamma = self.agent_config['gamma']
        
        # Thiết lập device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
                    if self.agent_config['device'] == 'auto' else torch.device(self.agent_config['device'])
        
        # Lấy input_shape và n_actions từ config
        self.input_shape = tuple(self.agent_config['input_shape'])
        self.n_actions = self.agent_config['n_actions']
        
        # Khởi tạo networks với config
        network_config = self.agent_config['network_architecture']
        self.policy_net = DQNNetwork(self.input_shape, self.n_actions, network_config).to(self.device)
        self.target_net = DQNNetwork(self.input_shape, self.n_actions, network_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Khởi tạo optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                lr=self.agent_config['learning_rate'])
        
        # Khởi tạo replay buffer
        self.memory = ReplayBuffer(self.agent_config['buffer_size'])
        
        # Khởi tạo các tham số exploration
        self.epsilon_start = self.agent_config['exploration_initial_eps']
        self.epsilon_final = self.agent_config['exploration_final_eps']
        self.epsilon_decay = self.agent_config['exploration_decay']
        self.epsilon = self.epsilon_start
        self.steps = 0
        
        # Tham số cập nhật target network
        self.target_update = self.agent_config['target_update_interval']
        
        # Tạo path dựa trên learning method
        method_name = self.learning_method.value
        self.model_dir = f"models/{method_name}_car_dodging"
        self.model_path = os.path.join(self.model_dir, "final_model.zip")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.model_dir, exist_ok=True)

    def select_action(self, state: np.ndarray) -> int:
        """Chọn action dựa trên policy epsilon-greedy"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
        else:
            return random.randrange(self.n_actions)
            
    def update(self) -> float:
        """Cập nhật model theo phương pháp đã chọn"""
        if self.learning_method == LearningMethod.DQN:
            return self._update_dqn()
        elif self.learning_method == LearningMethod.MONTE_CARLO:
            return self._update_monte_carlo()
        else:  # TD Learning
            return self._update_td()
            
    def _update_dqn(self) -> float:
        """Phương pháp DQN với Bellman equation"""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        # Bellman equation: Q(s,a) = r + γ * max(Q(s',a'))
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def _update_monte_carlo(self) -> float:
        """Monte Carlo learning với returns thực tế"""
        if not self.episode_buffer:
            return 0.0
            
        states, actions, rewards = zip(*self.episode_buffer)
        
        # Tính returns cho mỗi step
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
            
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Cập nhật Q values trực tiếp với returns
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(current_q_values, returns.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode buffer
        self.episode_buffer = []
        
        return loss.item()
        
    def _update_td(self) -> float:
        """TD Learning với TD target"""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        # TD target: r + γ * V(s')
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_state_values = self.policy_net(next_states).max(1)[0]
            td_targets = rewards + (1 - dones) * self.gamma * next_state_values
            
        loss = F.mse_loss(current_q_values, td_targets.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def add_to_memory(self, state, action, reward, next_state, done):
        """Thêm transition vào bộ nhớ phù hợp"""
        if self.learning_method == LearningMethod.MONTE_CARLO:
            self.episode_buffer.append((state, action, reward))
        else:
            self.memory.push(state, action, reward, next_state, done)
        
    def save(self, path: str):
        """Lưu model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        
    def load(self, path: str):
        """Tải model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
