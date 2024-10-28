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
        env = CarDodgingEnv()
        env = ImagePreprocessingWrapper(env)
        print(f"Đang tải model từ {model_path}")
        agent = DQN.load(model_path, env=env)
        print("Đã tải model thành công")
        return agent, env
    else:
        print(f"Không tìm thấy model tại {model_path}")
        return None, None

def main():

    # Tải agent
    model_path = "models/dqn_car_dodging/final_model.zip"
    agent, env = load_agent(model_path)

    if agent is not None:
        # Kiểm tra cấu trúc của model
        print(f"Model policy: {agent.policy}")
        # print(f"Observation space: {env.observation_space}")
        # print(f"Action space: {env.action_space}")
        
        mean_reward, std_reward = custom_evaluate_policy(agent, env, n_eval_episodes=5)
        # print(f"\nKết quả đánh giá sau {len(total_rewards)} episode:")
        print(f"Trung bình reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
