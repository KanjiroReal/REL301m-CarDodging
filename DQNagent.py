import os
import time
import warnings
import traceback
from tqdm import tqdm
import json

warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import pygame

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
    """Khởi tạo và thiết lập môi trường với preprocessing
    
    Args:
        config (dict, optional): Cấu hình cho môi trường. Default: None
    """
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
        last_minute = -1
        
        while True:
            # Xử lý sự kiện Pygame và render
            if render_training:
                env.render()
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
            else:
                # Vẫn check sự kiện khi không render
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

            current_time = time.time()
            elapsed_time = current_time - start_time
            elapsed_minutes = int(elapsed_time / 60)
            
            # Cập nhật progress bar theo thời gian
            if use_time_limit:
                # Cập nhật progress bar mỗi 0.1 giây
                if current_time - last_update >= 0.1:
                    pbar.n = min(int(elapsed_time), time_limit)
                    pbar.refresh()
                    last_update = current_time
                
                # Kiểm tra điều kiện dừng theo thời gian
                if elapsed_time >= time_limit:
                    print("\nĐã đạt giới hạn thời gian training")
                    break
            else:
                # Cập nhật theo số steps
                pbar.n = step
                pbar.refresh()
                
                # Kiểm tra điều kiện dừng theo steps
                if step >= total_timesteps:
                    print("\nĐã đạt số steps tối đa")
                    break
            
            # Training loop
            action = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            
            # Lấy thông tin từ info
            episode_time = info.get('elapsed_time', 0)
            current_score = info.get('current_reward', 0)
            
            current_episode_reward += reward
            
            # Cập nhật max survival time
            if episode_time > max_survival_time:
                max_survival_time = episode_time
            
            # Lưu transition và update model
            agent.add_to_memory(obs, action, reward, next_obs, done)
            if len(agent.memory) >= agent.batch_size:
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)
            
            if done:
                episode_count += 1
                total_rewards += current_episode_reward
                mean_loss = np.mean(episode_losses) if episode_losses else 0
                
                tqdm.write(f"Episode {episode_count} - Reward: {current_episode_reward}, Mean Loss: {mean_loss:.6f}")
                
                current_episode_reward = 0
                episode_losses = []
                obs, _ = env.reset()
            else:
                obs = next_obs
            
            # Cập nhật progress bar với thông tin mới
            pbar.set_postfix({
                'Episodes': episode_count,
                'Score': f'{current_score:.1f}',
                'Time': f'{episode_time:.1f}s',
                'Loss': f'{episode_losses[-1]:.6f}' if episode_losses else 'N/A'
            })
            
            step += 1
            
            # Lưu model mỗi phút
            if elapsed_minutes > last_minute:
                last_minute = elapsed_minutes
                agent.save(agent.model_path)
                
                mean_reward = total_rewards / max(episode_count, 1)
                mean_loss = np.mean(episode_losses) if episode_losses else 0
                
                tqdm.write("\n------------------------")
                tqdm.write(f"Đã train được: \n  - {elapsed_minutes} phút \n  - {episode_count} episodes \n  - {step} steps")
                tqdm.write(f"Thời gian sống tối đa: {max_survival_time:.1f}s")
                tqdm.write(f"Điểm trung bình: {mean_reward:.1f}")
                tqdm.write(f"Loss trung bình: {mean_loss:.6f}")
                tqdm.write(f"Đã lưu model tại: {agent.model_path}")
                tqdm.write("------------------------\n")

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