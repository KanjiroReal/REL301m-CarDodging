import os
import numpy as np
import pygame
from stable_baselines3 import DQN
from env import CarDodgingEnv
from DQNagent import ImagePreprocessingWrapper, custom_evaluate_policy
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_agent(model_path):
    """
    Tải agent từ đường dẫn đã cho
    """
    if os.path.exists(model_path):
        print(f"Đang tải model từ {model_path}")
        agent = DQN.load(model_path)
        print("Đã tải model thành công")
        return agent
    else:
        print(f"Không tìm thấy model tại {model_path}")
        return None


def play_episodes(agent, env, num_episodes=5):
    """
    Chơi và đánh giá agent trong số episode đã cho
    """
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Xử lý sự kiện Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        env.close()
                        return

            action, _ = agent.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            env.render()

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward}")

    return total_rewards


def main():
    # Khởi tạo môi trường
    env = CarDodgingEnv()
    env = ImagePreprocessingWrapper(env)

    # Tải agent
    model_path = "models/dqn_car_dodging/best_model.zip"
    agent = load_agent(model_path)

    if agent is not None:
        # Kiểm tra cấu trúc của model
        print(f"Model policy: {agent.policy}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Chơi và đánh giá
        total_rewards = play_episodes(agent, env)

        # Hiển thị kết quả
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"\nKết quả đánh giá sau {len(total_rewards)} episode:")
        print(f"Trung bình reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
