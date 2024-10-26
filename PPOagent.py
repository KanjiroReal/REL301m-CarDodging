from typing import Dict, Any
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from env import CarDodgingEnv

class VisualTrainingCallback(BaseCallback):
    """
    Callback để hiển thị, theo dõi và lưu mô hình trong quá trình huấn luyện
    """
    
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1, render_enabled: bool = True):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.render_enabled = render_enabled
        self.last_log_time = time.time()
        self.log_interval = 1.0  # Log mỗi 1 giây

        
    def _on_step(self) -> bool:
        try:
            # Render với xử lý lỗi
            if self.render_enabled:
                render_result = self.training_env.env_method("render")[0]
                if not render_result:
                    print("ERROR: Render Thất bại")
                    return False

            # Cập nhật phần thưởng hiện tại
            reward = self.locals['rewards'][0]
            self.current_episode_reward += reward
            
            # Xử lý khi episode kết thúc
            if self.locals['dones'][0]:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                
                # Log với tần suất giới hạn
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    if self.verbose > 0:
                        print(f"\nEpisode {self.episode_count}: Reward = {self.current_episode_reward:.0f}")
                    self.last_log_time = current_time
                
                self.current_episode_reward = 0
            
            # Kiểm tra và lưu mô hình tốt nhất
            if self.n_calls % self.check_freq == 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else -np.inf
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model'))
                    if self.verbose > 0:
                        print(f"Đã lưu agent với trung bình reward = {mean_reward:.2f} tại {self.save_path}.")
            # trả về true để tiếp tục train sau 1 step
            return True
            
        except Exception as e:
            print(f"Callback error: {str(e)}")
            return False  # dừng train khi có lỗi

def setup_env() -> gym.Env:
    """
    Khởi tạo và thiết lập môi trường

    Returns:
        gym.Env: Môi trường đã được thiết lập
    """
    env = CarDodgingEnv()
    return Monitor(env)


def create_ppo_agent(env: gym.Env, config: Dict[str, Any] = None) -> PPO:
    """
    Tạo agent PPO với cấu hình tùy chỉnh
    
    Args:
        env (gym.Env): Môi trường huấn luyện
        config (Dict[str, Any]): Cấu hình cho PPO
        
    Returns:
        PPO: Agent PPO đã được khởi tạo
    """
    if config is None:
        config = {
            'learning_rate': 0.0003,
            'n_steps': 1024,
            'batch_size': 64,
            'n_epochs': 2,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'verbose': 1,
            # Thêm các tham số để xử lý observation dạng hình ảnh
            'policy_kwargs': dict(
                features_extractor_kwargs=dict(features_dim=512),
                net_arch= dict(pi=[64, 64], vf=[64, 64])
            ),
        }
    
    return PPO('CnnPolicy', env, **config, device='auto')

def train_agent(agent: PPO, total_timesteps: int, save_path: str, 
                callback_freq: int = 1024) -> PPO:
    """
    Huấn luyện agent
    
    Args:
        agent (PPO): Agent cần huấn luyện
        total_timesteps (int): Tổng số bước huấn luyện
        save_path (str): Đường dẫn để lưu mô hình
        callback_freq (int): Tần suất kiểm tra và lưu mô hình
        
    Returns:
        PPO: Agent đã được huấn luyện
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Khởi tạo callback với chức năng hiển thị
    print("Bắt đầu khởi tạo callback...")
    callback = VisualTrainingCallback(
        check_freq=callback_freq,
        save_path=save_path
    )
    print("Hoàn thành khởi tạo callback")
    print("Bắt đầu huấn luyện agent...")
    try:
        # Huấn luyện agent
        agent.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True, reset_num_timesteps=False)
        
        # Lưu mô hình cuối cùng
        final_model_path = os.path.join(save_path, 'final_model')
        agent.save(final_model_path)
        print("Hoàn thành huấn luyện agent.")
        print(f"Agent đã được lưu tại {final_model_path}.")
    except KeyboardInterrupt:
        print("\nHuấn luyện đã bị dừng bởi người dùng.")
        # Lưu mô hình khi bị dừng
        interrupted_model_path = os.path.join(save_path, 'interrupted_model')
        agent.save(interrupted_model_path)
        print(f"Đã lưu agent tại: {interrupted_model_path}")
    
    return agent

def main() -> None:
    """
    Hàm chính để huấn luyện và kiểm tra agent
    """
    try:
        # Thiết lập môi trường với chế độ hiển thị
        env = setup_env()
        
        # Thiết lập thư mục lưu mô hình
        model_path = "models/ppo_car_dodging"
        
        # Tạo và huấn luyện agent
        agent = create_ppo_agent(env)
        agent = train_agent(
            agent,
            total_timesteps=5000,
            save_path=model_path
        )
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        # Đóng môi trường
        env.close()

if __name__ == "__main__":
    main()