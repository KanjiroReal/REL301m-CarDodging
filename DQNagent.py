from typing import Dict, Any
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space
from gymnasium import spaces
import cv2
import pygame

from env import CarDodgingEnv


class ImagePreprocessingWrapper(gym.ObservationWrapper):
    """
    Wrapper để xử lý và giảm kích thước ảnh observation
    """

    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, height, width),
            dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        obs = np.transpose(obs, (2, 0, 1))
        return obs


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
        self.log_interval = 1.0
        self.training_interrupted = False

    def _on_step(self) -> bool:
        try:
            # Xử lý sự kiện Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.training_interrupted = True
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.training_interrupted = True
                        return False

            if self.render_enabled:
                render_result = self.training_env.env_method("render")[0]
                if not render_result:
                    print("ERROR: Render Thất bại")
                    return False

            reward = self.locals['rewards'][0]
            self.current_episode_reward += reward

            if self.locals['dones'][0]:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)

                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    if self.verbose > 0:
                        print(f"Episode {self.episode_count}: Reward = {self.current_episode_reward:.0f}\n")
                    self.last_log_time = current_time

                self.current_episode_reward = 0

            if self.n_calls % self.check_freq == 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else -np.inf
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model'))
                    if self.verbose > 0:
                        print(f"Đã lưu agent với trung bình reward = {mean_reward:.2f} tại {self.save_path}.")

            return True

        except Exception as e:
            print(f"Callback error: {str(e)}")
            return False


def setup_env() -> gym.Env:
    """
    Khởi tạo và thiết lập môi trường với preprocessing
    """
    env = CarDodgingEnv()
    env = ImagePreprocessingWrapper(env)
    return Monitor(env)


def create_dqn_agent(env: gym.Env, config: Dict[str, Any] = None) -> DQN:
    """
    Tạo agent DQN với cấu hình tối ưu cho bộ nhớ
    """
    if config is None:
        config = {
            'learning_rate': 5e-4,  # Tốc độ học của mô hình
            'buffer_size': 50000,   # Kích thước của bộ nhớ replay buffer
            'learning_starts': 1000,  # Số bước trước khi bắt đầu học
            'batch_size': 64,  # Kích thước batch cho mỗi lần cập nhật
            'tau': 0.5,  # Hệ số cập nhật mạng target
            'gamma': 0.9,  # Hệ số chiết khấu cho phần thưởng trong tương lai
            'train_freq': 4,  # Tần suất cập nhật mạng (mỗi 4 bước)
            'gradient_steps': 2,  # Số bước gradient descent cho mỗi lần cập nhật
            'target_update_interval': 500,  # Tần suất cập nhật mạng target
            'exploration_fraction': 0.2,  # Phần thời gian dành cho khám phá
            'exploration_initial_eps': 1.0,  # Epsilon ban đầu cho khám phá
            'exploration_final_eps': 0.02,  # Epsilon cuối cùng cho khám phá
            'verbose': 1,  
            'policy_kwargs': dict(
                net_arch=[512, 256, 128, 64],  # Kiến trúc mạng neural (4 lớp ẩn)
                features_extractor_kwargs=dict(features_dim=256)  # Số chiều đầu ra của feature extractor
            ),
        }

    return DQN('CnnPolicy', env, **config, device='auto')


def train_agent(agent: DQN, total_timesteps: int, save_path: str,
                callback_freq: int = 1024, load_best: bool = False) -> DQN:
    """
    Huấn luyện agent
    """
    os.makedirs(save_path, exist_ok=True)

    best_model_path = os.path.join(save_path, 'final_model.zip')
    if load_best and os.path.exists(best_model_path):
        print(f"Đang tải best model từ {best_model_path}")
        agent = DQN.load(best_model_path, env=agent.env)
        print("Đã tải best model thành công")

    print("Bắt đầu khởi tạo callback...")
    callback = VisualTrainingCallback(
        check_freq=callback_freq,
        save_path=save_path
    )
    print("Hoàn thành khởi tạo callback")

    print("Bắt đầu huấn luyện agent...")
    try:
        agent.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

        final_model_path = os.path.join(save_path, 'final_model')
        agent.save(final_model_path)
        print("Hoàn thành huấn luyện agent.")
        print(f"Agent đã được lưu tại {final_model_path}.")

    except KeyboardInterrupt:
        print("\nHuấn luyện đã bị dừng bởi người dùng.")
        interrupted_model_path = os.path.join(save_path, 'interrupted_model')
        agent.save(interrupted_model_path)
        print(f"Đã lưu agent tại: {interrupted_model_path}")

    if callback.training_interrupted:
        print("\nHuấn luyện đã bị dừng thông qua giao diện Pygame.")
        interrupted_model_path = os.path.join(save_path, 'interrupted_model')
        agent.save(interrupted_model_path)
        print(f"Đã lưu agent tại: {interrupted_model_path}")

    return agent


def custom_evaluate_policy(model, env, n_eval_episodes=10):
    """
    Hàm evaluate tùy chỉnh với xử lý sự kiện Pygame
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Xử lý sự kiện Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return np.mean(episode_rewards), np.std(episode_rewards)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return np.mean(episode_rewards), np.std(episode_rewards)

            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            env.render()

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{n_eval_episodes}: Reward = {episode_reward}")

    return np.mean(episode_rewards), np.std(episode_rewards)


def evaluate_agent(agent: DQN, env: gym.Env, n_eval_episodes: int = 10) -> None:
    """
    Đánh giá hiệu suất của agent với xử lý sự kiện Pygame
    """
    mean_reward, std_reward = custom_evaluate_policy(agent, env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def main() -> None:
    """
    Hàm chính để huấn luyện và kiểm tra agent
    """
    train_timesteps = 10000

    try:
        env = setup_env()
        model_path = "models/dqn_car_dodging"

        agent = create_dqn_agent(env)
        agent = train_agent(
            agent,
            total_timesteps=train_timesteps,
            save_path=model_path,
            load_best=True
        )

        # print("\nĐánh giá agent...")
        # evaluate_agent(agent, env)

    except Exception as e:
        print(f"Lỗi: {str(e)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()