import os
import time
import warnings
import traceback
import json

warnings.filterwarnings("ignore", category=UserWarning)

import pygame
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from env import CarDodgingEnv
from models import DQNAgent


def setup_env(config) -> gym.Env:
    """Khởi tạo và thiết lập môi trường"""
    if config is None:
        raise ValueError("Config không được để trống")
        
    required_keys = ['num_lanes', 'agent_limit_tick', 'spawn_interval']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Thiếu các tham số cấu hình: {missing_keys}")
        
    env = CarDodgingEnv(config)
    return env

def _handle_pygame_events(env, render_training):
    """Xử lý các sự kiện Pygame"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT or \
           (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            return True, render_training
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            render_training = not render_training
            tqdm.write(f"\nĐã {'bật' if render_training else 'tắt'} render")
    return False, render_training

def _update_metrics(metrics, reward, loss, elapsed_time):
    """Cập nhật các metrics training"""
    metrics['episode_reward'] += reward
    if loss is not None:
        metrics['episode_losses'].append(loss)
    metrics['current_time'] = elapsed_time
    return metrics

def train_agent(agent: DQNAgent, env: gym.Env, total_timesteps: int) -> DQNAgent:
    """Huấn luyện agent"""
    try:
        # Load model nếu có
        if os.path.exists(agent.model_path):
            print(f"\nĐang tải model từ {agent.model_path}")
            agent.load(agent.model_path)
        
        # Khởi tạo các biến tracking
        metrics = {
            'episode_count': 0,
            'total_rewards': 0,
            'episode_reward': 0,
            'episode_losses': [],
            'max_survival_time': 0,
            'current_time': 0,
            'steps_per_episode': []
        }
        
        obs, _ = env.reset()
        step = 0
        episode_steps = 0
        training_start_time = time.time()
        last_save_time = training_start_time
        last_update = training_start_time
        last_frame_time = training_start_time
        
        # Config
        game_fps = env.metadata.get("render_fps", 60)
        frame_time = 1.0 / game_fps
        use_time_limit = agent.training_config.get('use_time_limit', False)
        training_minutes = agent.training_config.get('training_minutes', 60)
        render_training = agent.training_config.get('render_training', True)
        time_limit = training_minutes * 60
        
        with tqdm(total=time_limit if use_time_limit else total_timesteps,
                desc='Training Progress', 
                unit='s' if use_time_limit else ' steps') as pbar:
            
            while True:
                current_time = time.time()
                elapsed_time = current_time - training_start_time
                
                # Xử lý Pygame events
                should_quit, render_training = _handle_pygame_events(env, render_training)
                if should_quit:
                    print("\nDừng training theo yêu cầu người dùng")
                    break
                
                # Đồng bộ frame rate
                if current_time - last_frame_time >= frame_time:
                    if render_training:
                        env.render()

                    # Training step
                    action = agent.select_action(obs)
                    next_obs, reward, done, _, info = env.step(action)
                    step += 1
                    episode_steps += 1
                    last_frame_time = current_time
                    
                    # Update memory và train
                    agent.add_to_memory(obs, action, reward, next_obs, done)
                    loss = None
                    if len(agent.memory) >= agent.batch_size * 2:
                        loss = agent.update()
                    
                    # Cập nhật metrics
                    metrics = _update_metrics(metrics, reward, loss, elapsed_time)
                    metrics['max_survival_time'] = max(metrics['max_survival_time'], 
                                                     info.get('elapsed_time', 0))
                    
                    # Xử lý kết thúc episode
                    if done:
                        metrics['episode_count'] += 1
                        metrics['total_rewards'] += metrics['episode_reward']
                        metrics['steps_per_episode'].append(episode_steps)
                        
                        tqdm.write(f"Episode {metrics['episode_count']} - "
                                 f"Reward: {metrics['episode_reward']:.2f} - "
                                 f"Steps: {episode_steps}")
                        
                        metrics['episode_reward'] = 0
                        episode_steps = 0
                        obs, _ = env.reset()
                    else:
                        obs = next_obs
                
                # Cập nhật progress bar
                if current_time - last_update >= 1:
                    if use_time_limit:
                        actual_update = min(int(current_time - last_update), 
                                         time_limit - pbar.n)
                        if actual_update > 0:
                            pbar.update(actual_update)
                    else:
                        steps_since_update = step - pbar.n
                        if steps_since_update > 0:
                            pbar.update(steps_since_update)
                    
                    avg_steps = np.mean(metrics['steps_per_episode']) if metrics['steps_per_episode'] else 0
                    pbar.set_postfix({
                        'Episodes': metrics['episode_count'],
                        'Score': f"{metrics['episode_reward']:.1f}",
                        'Avg Steps': f"{avg_steps:.1f}",
                        'Epsilon': f"{agent.epsilon:.3f}"
                    })
                    last_update = current_time

                # Kiểm tra điều kiện dừng
                if use_time_limit and elapsed_time >= time_limit:
                    print(f"\nĐã hoàn thành training sau {int(elapsed_time)} giây!")
                    break
                elif not use_time_limit and step >= total_timesteps:
                    print(f"\nĐã hoàn thành training sau {int(elapsed_time)} giây!")
                    break
                
                # Lưu model định kỳ
                if current_time - last_save_time >= 60:
                    last_save_time = current_time
                    agent.save(agent.model_path)
                    _log_training_stats(metrics, step, agent.model_path)

        # Lưu model cuối cùng
        agent.save(agent.model_path)
        print(f"Đã lưu model tại: {agent.model_path}")
        return agent

    except Exception as e:
        print(f"\nLỗi trong quá trình training: {str(e)}")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        return agent

def _log_training_stats(metrics, step, model_path):
    """Log thống kê training"""
    mean_reward = metrics['total_rewards'] / max(metrics['episode_count'], 1)
    
    tqdm.write("\n------------------------")
    tqdm.write(f"Đã train được: \n  - {metrics['episode_count']} episodes \n  - {step} steps")
    tqdm.write(f"Điểm trung bình: {mean_reward:.1f}")
    tqdm.write(f"Thời gian sống lâu nhất: {metrics['max_survival_time']:.1f}s")
    tqdm.write(f"Đã lưu model tại: {model_path}")
    tqdm.write("------------------------\n")

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