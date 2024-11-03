import os
import time
import warnings
import traceback
import json

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tqdm import tqdm
from env import CarDodgingEnv
from models import DQNAgent


class ImagePreprocessingWrapper(gym.ObservationWrapper):
    """Wrapper để xử lý và giảm kích thước ảnh observation"""
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


def setup_env(config) -> gym.Env:
    """Khởi tạo và thiết lập môi trường với preprocessing"""
    if config is None:
        raise ValueError("Config không được để trống")
        
    required_keys = ['num_lanes', 'agent_limit_tick', 'spawn_interval']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Thiếu các tham số cấu hình: {missing_keys}")
        
    env = CarDodgingEnv(config)
    env = ImagePreprocessingWrapper(env)
    return env

def train_agent(agent: DQNAgent, env: gym.Env, total_timesteps: int) -> DQNAgent:
    """Huấn luyện agent"""
    try:
        # Load model trước khi train nếu có
        if os.path.exists(agent.model_path):
            print(f"\nĐang tải model từ {agent.model_path}")
            agent.load(agent.model_path)
            print("Đã tải model thành công")
        
        obs, _ = env.reset()
        step = 0
        start_time = time.time()
        last_update = start_time
        episode_count = 0
        total_rewards = 0
        current_episode_reward = 0
        episode_losses = []
        max_survival_time = 0
        
        # Lấy time limit và render config từ config
        use_time_limit = agent.training_config.get('use_time_limit', False)
        training_minutes = agent.training_config.get('training_minutes', 60)
        render_training = agent.training_config.get('render_training', True)
        time_limit = training_minutes * 60
        
        # Khởi tạo progress bar
        pbar = tqdm(total=time_limit if use_time_limit else total_timesteps,
                   desc='Training Progress',
                   unit='s' if use_time_limit else ' steps')
        
        while True:
            # Xử lý sự kiện Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    print("\nDừng training theo yêu cầu người dùng")
                    agent.save(agent.model_path)
                    print(f"Đã lưu model tại: {agent.model_path}")
                    return agent
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    render_training = not render_training
                    tqdm.write(f"\nĐã {'bật' if render_training else 'tắt'} render")
            
            if render_training:
                env.render()

            # Training loop
            action = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            current_time = time.time()
            
            # Cập nhật progress bar mỗi 1s
            if current_time - last_update >= 1:
                if use_time_limit:
                    pbar.update(int(current_time - last_update))
                else:
                    pbar.update(1)
                pbar.set_postfix({
                    'Episodes': episode_count,
                    'Score': f'{current_episode_reward:.1f}',
                    'Loss': f'{episode_losses[-1]:.6f}' if episode_losses else 'N/A',
                    'Epsilon': f'{agent.epsilon:.3f}'
                })
                last_update = current_time
                
            current_episode_reward += reward
            
            if done:
                episode_count += 1
                total_rewards += current_episode_reward
                mean_loss = np.mean(episode_losses) if episode_losses else 0
                
                tqdm.write(f"Episode {episode_count} - Reward: {current_episode_reward:.2f}, Mean Loss: {mean_loss:.6f}")
                
                current_episode_reward = 0
                episode_losses = []
                obs, _ = env.reset()
            else:
                obs = next_obs
                
            # Cập nhật max survival time
            if info.get('elapsed_time', 0) > max_survival_time:
                max_survival_time = info.get('elapsed_time', 0)
            
            # Lưu transition và update model
            agent.add_to_memory(obs, action, reward, next_obs, done)
            if len(agent.memory) >= agent.batch_size * 2:
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)
            
            # Lưu model mỗi phút
            if current_time - start_time >= 60:
                start_time = current_time
                agent.save(agent.model_path)
                
                mean_reward = total_rewards / max(episode_count, 1)
                mean_loss = np.mean(episode_losses) if episode_losses else 0
                
                tqdm.write("\n------------------------")
                tqdm.write(f"Đã train được: \n  - {int((current_time - start_time) / 60)} phút \n  - {episode_count} episodes \n  - {step} steps")
                tqdm.write(f"Thời gian sống tối đa: {max_survival_time:.1f}s")
                tqdm.write(f"Điểm trung bình: {mean_reward:.1f}")
                tqdm.write(f"Loss trung bình: {mean_loss:.6f}")
                tqdm.write(f"Đã lưu model tại: {agent.model_path}")
                tqdm.write("------------------------\n")

            step += 1
            elapsed_time = time.time() - start_time
            
            
            if (use_time_limit and elapsed_time >= time_limit) or \
               (not use_time_limit and step >= total_timesteps):
                break

        # Lưu model cuối cùng và return
        agent.save(agent.model_path)
        print(f"Đã lưu model tại: {agent.model_path}")
        return agent

    except Exception as e:
        print(f"\nLỗi trong quá trình training: {str(e)}")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    finally:
        if 'pbar' in locals():
            pbar.close()
        
    return agent

def main() -> None:
    """Hàm chính để huấn luyện và kiểm tra agent"""
    config_path = "config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        env = setup_env(config.get('env_config'))
        agent = DQNAgent(config_path)
        
        # Train agent
        agent = train_agent(
            agent,
            env,
            total_timesteps=agent.training_config['total_timesteps']
        )

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    finally:
        env.close()

if __name__ == "__main__":
    main()