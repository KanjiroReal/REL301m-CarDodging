# REL301m-CarDodging
## --- [ENGLISH BELOW](#rel301m-cardodging-1) ---


Dự án này triển khai một môi trường game đơn giản "Car Dodging" và huấn luyện các agent AI để chơi game sử dụng thuật toán học tăng cường.

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/your-username/REL301m-CarDodging.git
cd REL301m-CarDodging
```

2. Tạo và kích hoạt môi trường ảo:

```
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

3. Cài đặt các thư viện cần thiết:

```
pip install -r requirements.txt
```

## Cấu trúc dự án

- `env.py`: Định nghĩa môi trường CarDodgingEnv
- `DQNagent.py`: Triển khai agent sử dụng thuật toán DQN
- `PPOagent.py`: Triển khai agent sử dụng thuật toán PPO
- `play.py`: Script để chạy và đánh giá agent đã huấn luyện

## Chỉnh sửa môi trường

Để điều chỉnh môi trường, mở file `env.py` và chỉnh sửa các tham số như:

- `NUM_LANES`: Số làn đường
- `AGENT_LIMIT_TICK`: Thời gian giới hạn giữa các hành động của agent
- `PENALTY`: Điểm phạt khi va chạm
- `REWARD`: Phần thưởng cho mỗi bước di chuyển
- `SPAWN_INTERVAL`: Khoảng thời gian giữa các lần sinh obstacle
- `MAX_STEPS_PER_EPISODE`: Số bước tối đa cho mỗi episode
- `OBSTACLE_SPEED`: Tốc độ di chuyển của obstacle

## Huấn luyện agent

### Sử dụng DQN

1. Mở file `DQNagent.py`
2. Điều chỉnh các tham số cấu hình trong hàm `create_dqn_agent()`
3. Chạy script để huấn luyện:

```
python DQNagent.py
```

### Sử dụng PPO

1. Mở file `PPOagent.py`
2. Điều chỉnh các tham số cấu hình trong hàm `create_ppo_agent()`
3. Chạy script để huấn luyện:

```
python PPOagent.py
```

## Chạy và đánh giá agent

Để chạy và đánh giá agent đã huấn luyện:

1. Đảm bảo bạn đã huấn luyện và lưu model (thường trong thư mục `models/`)
2. Mở file `play.py` và điều chỉnh đường dẫn đến model trong biến `model_path`
3. Chạy script:

```
python play.py
```

Script này sẽ tải model đã huấn luyện và chạy một số episode để đánh giá hiệu suất của agent.

## Tùy chỉnh

- Để thay đổi số episode đánh giá, điều chỉnh tham số `num_episodes` trong hàm `play_episodes()` của file `play.py`
- Để điều chỉnh cách hiển thị kết quả, chỉnh sửa hàm `main()` trong `play.py`

## Lưu ý

- Đảm bảo rằng bạn đã cài đặt đầy đủ các thư viện cần thiết trước khi chạy các script
- Nếu gặp vấn đề với việc render môi trường, hãy kiểm tra cài đặt Pygame của bạn

---

# REL301m-CarDodging

This project implements a simple "Car Dodging" game environment and trains AI agents to play the game using reinforcement learning algorithms.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/REL301m-CarDodging.git
cd REL301m-CarDodging
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required libraries:

```
pip install -r requirements.txt
```

## Project Structure

- `env.py`: Defines the CarDodgingEnv environment
- `DQNagent.py`: Implements the agent using the DQN algorithm
- `PPOagent.py`: Implements the agent using the PPO algorithm
- `play.py`: Script to run and evaluate the trained agent

## Modifying the Environment

To adjust the environment, open the `env.py` file and modify parameters such as:

- `NUM_LANES`: Number of lanes
- `AGENT_LIMIT_TICK`: Time limit between agent actions
- `PENALTY`: Penalty points for collisions
- `REWARD`: Reward for each movement step
- `SPAWN_INTERVAL`: Time interval between obstacle spawns
- `MAX_STEPS_PER_EPISODE`: Maximum number of steps per episode
- `OBSTACLE_SPEED`: Speed of obstacle movement

## Training the Agent

### Using DQN

1. Open the `DQNagent.py` file
2. Adjust the configuration parameters in the `create_dqn_agent()` function
3. Run the script to train:

```
python DQNagent.py
```

### Using PPO

1. Open the `PPOagent.py` file
2. Adjust the configuration parameters in the `create_ppo_agent()` function
3. Run the script to train:

```
python PPOagent.py
```

## Running and Evaluating the Agent

To run and evaluate the trained agent:

1. Ensure you have trained and saved the model (usually in the `models/` directory)
2. Open the `play.py` file and adjust the path to the model in the `model_path` variable
3. Run the script:

```
python play.py
```

This script will load the trained model and run a number of episodes to evaluate the agent's performance.

## Customization

- To change the number of evaluation episodes, adjust the `num_episodes` parameter in the `play_episodes()` function of the `play.py` file
- To adjust how results are displayed, edit the `main()` function in `play.py`

## Notes

- Make sure you have installed all the necessary libraries before running the scripts
- If you encounter issues with rendering the environment, check your Pygame installation
