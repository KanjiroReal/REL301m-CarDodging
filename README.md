# REL301m-CarDodging
## --- [ENGLISH BELOW](#rel301m-cardodging-1) ---

Dự án này triển khai một môi trường game đơn giản "Car Dodging" và huấn luyện các agent AI để chơi game sử dụng các phương pháp học tăng cường khác nhau (DQN, Monte Carlo, TD Learning).

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/your-username/REL301m-CarDodging.git
cd REL301m-CarDodging
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

- `env.py`: Định nghĩa môi trường CarDodgingEnv
- `models.py`: Triển khai các model DQN và các phương pháp học tăng cường
- `DQNagent.py`: Script huấn luyện agent
- `play.py`: Script để chạy và đánh giá agent đã huấn luyện
- `config.json`: File cấu hình cho môi trường và agent
- `images/`: Thư mục chứa hình ảnh cho game
  - `agents/`: Hình ảnh xe của agent (agent.png)
  - `obstacles/`: Hình ảnh các xe vật cản (car1.png đến car12.png)

## Cấu hình

Tất cả cấu hình được định nghĩa trong file `config.json`, bao gồm:

### Cấu hình môi trường (env_config)
- Số làn đường, kích thước cửa sổ
- Tốc độ game, reward/penalty
- Kích thước xe, màu sắc đường và vạch kẻ đường
- Tốc độ vật cản và tần suất xuất hiện

### Cấu hình agent (agent_config)
- Tham số học tập (learning rate, gamma, ...)
- Cấu trúc mạng neural (CNN layers, FC layers)
- Tham số exploration và buffer size

### Cấu hình training (training_config)
- Số bước training hoặc thời gian training
- Phương pháp học (DQN/Monte Carlo/TD)
- Tần suất cập nhật và lưu model

### Cấu hình chơi (play_config)
- Phương pháp học để load model
- Số episode đánh giá

## Huấn luyện agent

1. Điều chỉnh cấu hình trong `config.json`:
   - Chọn phương pháp học trong `training_config.learning_method`
   - Điều chỉnh thời gian hoặc số bước training
   - Tùy chỉnh các tham số khác

2. Chạy script huấn luyện:
```bash
python DQNagent.py
```

Trong quá trình training:
- Nhấn Q để dừng training và lưu model
- Nhấn R để bật/tắt render
- Progress bar hiển thị tiến trình
- Thống kê được in mỗi phút
- Model được tự động lưu mỗi phút và khi kết thúc

## Chạy và đánh giá agent

1. Chọn phương pháp học trong `play_config.learning_method`
2. Chạy script đánh giá:
```bash
python play.py
```

Script sẽ:
- Tự động tải model tương ứng với phương pháp học đã chọn
- Chạy số episode đánh giá được cấu hình
- Hiển thị thống kê hiệu suất và thông tin model

## Cấu trúc thư mục models

```
models/
  ├── dqn_car_dodging/
  │   └── final_model.zip
  ├── monte_carlo_car_dodging/
  │   └── final_model.zip
  └── td_car_dodging/
      └── final_model.zip
```

## Lưu ý

- Đảm bảo các thư mục `models/` tồn tại trước khi training
- Model được lưu tự động theo phương pháp học
- Có thể tiếp tục training từ model đã lưu
- Cần có GPU để training hiệu quả (tự động phát hiện qua config)

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
- `models.py`: Implements the DQN and reinforcement learning algorithms
- `DQNagent.py`: Script to train the agent
- `play.py`: Script to run and evaluate the trained agent
- `config.json`: Configuration file for the environment and agent
- `images/`: Directory containing images for the game
  - `agents/`: Images of the agent's car
  - `obstacles/`: Images of the obstacles

## Configuration

All configuration is defined in the `config.json` file, which includes:

### Environment Configuration (env_config)
- Number of lanes and window size
- Game speed and reward/penalty
- Car size and lane color

### Agent Configuration (agent_config)
- Learning rate, gamma, etc.
- Network architecture (CNN layers, FC layers)
- Exploration parameters

### Training Configuration (training_config)
- Number of training steps
- Learning method (DQN/Monte Carlo/TD)
- Other training parameters

## Training the Agent

1. Adjust the configuration in the `config.json` file as needed
2. Run the training script:
```bash
python DQNagent.py
```

During training:
- Press Q to stop training and save the model
- Progress bar shows training progress
- Statistics are printed every minute

## Running and Evaluating the Agent

1. Ensure you have trained and saved the model (usually in the `models/dqn_car_dodging/`)
2. Run the evaluation script:
```bash
python play.py
```

This script will:
- Load the trained model
- Run 5 evaluation episodes
- Display performance statistics

## Customization

- Change the network architecture: Edit the `network_architecture` in the config
- Change the learning method: Adjust the `learning_method` in the config
- Adjust the environment: Modify the parameters in the `env_config`

## Notes

- Ensure the `models/` and `logs/` directories exist before training
- Verify the paths in the config are correct for the directory structure
- A GPU is recommended for efficient training (automatically detected via config)
