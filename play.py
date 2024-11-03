import os
import numpy as np
import pygame
import json
import traceback
from env import CarDodgingEnv
from models import DQNAgent
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_agent(config_path: str):
    """
    Tải agent từ config và model đã lưu
    
    Args:
        config_path: Đường dẫn đến file config
        
    Returns:
        tuple: (agent, env) hoặc (None, None) nếu có lỗi
    """
    try:
        if not os.path.exists(config_path):
            print(f"Không tìm thấy config tại {config_path}")
            return None, None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Lấy phương pháp học từ play_config
        learning_method = config.get('play_config', {}).get('learning_method', 'dqn')
        model_path = f"models/{learning_method}_car_dodging/final_model.zip"
        
        if not os.path.exists(model_path):
            print(f"Không tìm thấy model cho phương pháp {learning_method} tại {model_path}")
            return None, None
            
        # Khởi tạo môi trường với config
        env = CarDodgingEnv(config.get('env_config'))
        
        # Khởi tạo agent với config
        print(f"Đang tải config từ {config_path}")
        agent = DQNAgent(config_path)
        
        # Load model weights
        print(f"Đang tải model {learning_method} từ {model_path}")
        agent.load(model_path)
        print("Đã tải model thành công")
        
        # Tắt exploration khi chơi
        agent.epsilon = 0.0
        
        return agent, env
        
    except Exception as e:
        print(f"Lỗi khi tải agent: {str(e)}")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        return None, None

def play_episode(agent: DQNAgent, env: CarDodgingEnv, render: bool = True) -> tuple:
    """
    Chơi một episode
    
    Returns:
        tuple: (total_reward, survival_time)
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0
    survival_time = 0
    
    while not done:
        # Xử lý sự kiện Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return total_reward, survival_time
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return total_reward, survival_time
        
        # Chọn và thực hiện action
        action = agent.select_action(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        
        # Lấy thời gian sống từ info
        survival_time = info.get('elapsed_time', 0)
        
        if render:
            env.render()
            
    return total_reward, survival_time

def evaluate_agent(agent: DQNAgent, env: CarDodgingEnv, n_episodes: int = 5) -> tuple:
    rewards = []
    survival_times = []
    
    for i in range(n_episodes):
        reward, time = play_episode(agent, env)
        rewards.append(reward)
        survival_times.append(time)
        print(f"Episode {i+1}/{n_episodes}: Score = {reward:.1f}, Time = {time:.1f}s")
        
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_time = np.mean(survival_times)
    std_time = np.std(survival_times)
    
    return mean_reward, std_reward, mean_time, std_time

def main():
    config_path = "config.json"
    
    # Tải agent theo phương pháp học được chọn
    agent, env = load_agent(config_path)
    
    if agent is not None:
        try:
            # Load số episode đánh giá từ config
            with open(config_path, 'r') as f:
                config = json.load(f)
            n_episodes = config.get('play_config', {}).get('n_evaluation_episodes', 5)
            
            # In thông tin cấu trúc mạng và phương pháp học
            print("\nThông tin model:")
            print(f"Phương pháp học: {agent.learning_method.value}")
            print(f"Policy network: {agent.policy_net}")
            print(f"Device: {agent.device}")
            
            # Đánh giá agent
            print(f"\nBắt đầu đánh giá agent ({n_episodes} episodes)...")
            mean_reward, std_reward, mean_time, std_time = evaluate_agent(agent, env, n_episodes=n_episodes)
            print(f"\nKết quả đánh giá:")
            print(f"Điểm trung bình: {mean_reward:.1f} ± {std_reward:.1f}")
            print(f"Thời gian sống: {mean_time:.1f}s ± {std_time:.1f}s")
            
        except Exception as e:
            print(f"Lỗi khi chạy agent: {str(e)}")
            print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        finally:
            env.close()

if __name__ == "__main__":
    main()
