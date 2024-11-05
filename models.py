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
    """Mạng neural cho DQN với input dạng vector"""
    def __init__(self, input_shape: dict, n_actions: int, network_config: dict):
        super().__init__()
        
        # Tính toán kích thước input flattened
        self.input_size = (
            1 +  # agent_lane
            3 * input_shape['num_lanes'] +  # obstacles_info (presence, distances, speeds)
            2 +  # game_info (elapsed_time, current_speed)
            2    # movement_info (last_direction, unnecessary_moves)
        )
        
        layers = []
        current_size = self.input_size
        
        # Bỏ batch norm ở input layer
        for size in network_config['fc_layers']:
            layers.extend([
                nn.Linear(current_size, size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_size = size
        
        # Output layer
        layers.append(nn.Linear(current_size, n_actions))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo weights với Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, x: dict) -> torch.Tensor:
        # Đảm bảo tất cả tensors có cùng số chiều
        game_info_elapsed = x['game_info']['elapsed_time'].squeeze(-1) if x['game_info']['elapsed_time'].dim() == 3 else x['game_info']['elapsed_time']
        game_info_speed = x['game_info']['current_speed'].squeeze(-1) if x['game_info']['current_speed'].dim() == 3 else x['game_info']['current_speed']
        movement_last_dir = x['movement_info']['last_direction'].squeeze(-1) if x['movement_info']['last_direction'].dim() == 3 else x['movement_info']['last_direction']
        movement_unnecessary = x['movement_info']['unnecessary_moves'].squeeze(-1) if x['movement_info']['unnecessary_moves'].dim() == 3 else x['movement_info']['unnecessary_moves']
    
        flattened = torch.cat([
            x['agent_lane'],
            x['obstacles_info']['presence'],
            x['obstacles_info']['distances'],
            x['obstacles_info']['speeds'],
            game_info_elapsed,
            game_info_speed,
            movement_last_dir,
            movement_unnecessary
        ], dim=1)
    
        # Xử lý NaN và Inf
        flattened = torch.where(torch.isnan(flattened), torch.zeros_like(flattened), flattened)
        flattened = torch.clamp(flattened, -1e6, 1e6)  # Giới hạn giá trị
    
        output = self.network(flattened)
    
        # Normalize output để tránh Q-values quá lớn
        output = output / output.abs().max()
    
        return output

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
        # Đọc config t file JSON
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.agent_config = config['agent_config']
        self.training_config = config['training_config']
        
        self.learning_method = LearningMethod(self.training_config['learning_method'])
        self.episode_buffer = []
        self.batch_size = self.agent_config['batch_size']
        self.gamma = self.agent_config['gamma']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
                    if self.agent_config['device'] == 'auto' else torch.device(self.agent_config['device'])
        
        # Thay đổi input_shape thành dict chứa thông tin về kích thước observation
        self.input_shape = {
            'num_lanes': self.agent_config['num_lanes']
        }
        
        self.n_actions = self.agent_config['n_actions']
        
        network_config = self.agent_config['network_architecture']
        self.policy_net = DQNNetwork(self.input_shape, self.n_actions, network_config).to(self.device)
        self.target_net = DQNNetwork(self.input_shape, self.n_actions, network_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Khởi tạo optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                lr=self.agent_config['learning_rate'])
        
        # Khởi tạo replay buffer
        self.memory = ReplayBuffer(self.agent_config['buffer_size'])
        
        # Tham số cập nhật target network
        self.target_update = self.agent_config['target_update_interval']
        
        # Tạo path dựa trên learning method
        method_name = self.learning_method.value
        self.model_dir = f"models/{method_name}_car_dodging"
        self.model_path = os.path.join(self.model_dir, "final_model.zip")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.steps = 0

    def select_action(self, state: dict) -> int:
        """Chọn action dựa trên Q-values"""
        with torch.no_grad():
            # Chuyển state thành tensors
            state_tensors = {
                'agent_lane': torch.FloatTensor(state['agent_lane']).unsqueeze(0).to(self.device),
                'obstacles_info': {
                    'presence': torch.FloatTensor(state['obstacles_info']['presence']).unsqueeze(0).to(self.device),
                    'distances': torch.FloatTensor(state['obstacles_info']['distances']).unsqueeze(0).to(self.device),
                    'speeds': torch.FloatTensor(state['obstacles_info']['speeds']).unsqueeze(0).to(self.device)
                },
                'game_info': {
                    'elapsed_time': torch.FloatTensor([state['game_info']['elapsed_time']]).unsqueeze(0).to(self.device),
                    'current_speed': torch.FloatTensor([state['game_info']['current_speed']]).unsqueeze(0).to(self.device)
                },
                'movement_info': {
                    'last_direction': torch.FloatTensor([state['movement_info']['last_direction']]).unsqueeze(0).to(self.device),
                    'unnecessary_moves': torch.FloatTensor([state['movement_info']['unnecessary_moves']]).unsqueeze(0).to(self.device)
                }
            }
            
            # Tính Q-values và normalize
            q_values = self.policy_net(state_tensors)
            
            # Thêm noise nhỏ để tránh các hành động giống nhau
            noise = torch.randn_like(q_values) * 0.1
            q_values = q_values + noise
            # print(q_values)
            # print(q_values.max(1)[1].item())
            return q_values.max(1)[1].item()
            
    def update(self) -> float:
        """Cập nhật model theo phương pháp đã chọn"""
        if self.learning_method == LearningMethod.DQN:
            return self._update_dqn()
        elif self.learning_method == LearningMethod.MONTE_CARLO:
            return self._update_monte_carlo()
        else: 
            return self._update_td()
            
    def _process_batch_states(self, states) -> dict:
        """Chuyển đổi batch states thành tensors"""
        return {
            'agent_lane': torch.FloatTensor(np.vstack([s['agent_lane'] for s in states])).to(self.device),
            'obstacles_info': {
                'presence': torch.FloatTensor(np.vstack([s['obstacles_info']['presence'] for s in states])).to(self.device),
                'distances': torch.FloatTensor(np.vstack([s['obstacles_info']['distances'] for s in states])).to(self.device),
                'speeds': torch.FloatTensor(np.vstack([s['obstacles_info']['speeds'] for s in states])).to(self.device)
            },
            'game_info': {
                'elapsed_time': torch.FloatTensor(np.vstack([s['game_info']['elapsed_time'] for s in states])).to(self.device),
                'current_speed': torch.FloatTensor(np.vstack([s['game_info']['current_speed'] for s in states])).to(self.device)
            },
            'movement_info': {
                'last_direction': torch.FloatTensor(np.vstack([s['movement_info']['last_direction'] for s in states])).to(self.device),
                'unnecessary_moves': torch.FloatTensor(np.vstack([s['movement_info']['unnecessary_moves'] for s in states])).to(self.device)
            }
        }

    def _normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Chuẩn hóa giá trị"""
        return (values - values.mean()) / (values.std() + 1e-7)

    def _update_networks(self, loss: torch.Tensor) -> float:
        """Cập nhật networks"""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def _update_dqn(self) -> float:
        """Phương pháp DQN với Bellman equation"""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Chuyển đổi states và next_states
        batch_states = self._process_batch_states(batch[0])
        batch_next_states = self._process_batch_states(batch[3])
        
        # Chuyển đổi các thành phần khác
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = self._normalize_values(torch.FloatTensor(batch[2]).to(self.device))
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        # Tính toán Q-values
        current_q_values = self.policy_net(batch_states).gather(1, actions.unsqueeze(1))
        
        # Tính toán Q-values mục tiêu với target network
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        return self._update_networks(loss)

    def _update_td(self) -> float:
        """TD Learning với TD target"""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Chuyển đổi states và next_states
        batch_states = self._process_batch_states(batch[0])
        batch_next_states = self._process_batch_states(batch[3])
        
        # Chuyển đổi các thành phần khác
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = self._normalize_values(torch.FloatTensor(batch[2]).to(self.device))
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        # Tính TD target
        current_q_values = self.policy_net(batch_states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_state_values = self.policy_net(batch_next_states).max(1)[0]
            td_targets = rewards + (1 - dones) * self.gamma * next_state_values
            
        loss = F.smooth_l1_loss(current_q_values, td_targets.unsqueeze(1))
        return self._update_networks(loss)

    def _update_monte_carlo(self) -> float:
        """Monte Carlo learning với returns thực tế"""
        if not self.episode_buffer:
            return 0.0
            
        states, actions, rewards = zip(*self.episode_buffer)
        
        # Chuyển đổi states
        batch_states = self._process_batch_states(states)
        
        # Tính returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = self._normalize_values(torch.FloatTensor(returns).to(self.device))
        
        # Chuyển đổi actions
        actions = torch.LongTensor(actions).to(self.device)
        
        # Tính loss
        current_q_values = self.policy_net(batch_states).gather(1, actions.unsqueeze(1))
        loss = F.smooth_l1_loss(current_q_values, returns.unsqueeze(1))
        
        # Reset episode buffer
        self.episode_buffer = []
        
        return self._update_networks(loss)

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
            'steps': self.steps
        }, path)
        
    def load(self, path: str):
        """Tải model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']

