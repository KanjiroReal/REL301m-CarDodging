# Car Dodging AI

[en English](README.md)

Dự án này triển khai một môi trường game đơn giản "Car Dodging" và huấn luyện các agent AI để chơi game sử dụng các phương pháp học tăng cường khác nhau (DQN, Monte Carlo, TD Learning).

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/KanjiroReal/REL301m-CarDodging.git
cd REL301m-CarDodging
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

- `env.py`: Định nghĩa môi trường CarDodgingEnv với các tính năng:
  - Nhiều làn đường có thể cấu hình
  - Hệ thống reward phức tạp (survival, dodge, movement)
  - Hiển thị thông tin chi tiết (score, time, dodge zones)
  - Hỗ trợ nhiều loại xe vật cản khác nhau

- `models.py`: Triển khai các model và phương pháp học:
  - DQN với replay buffer
  - Monte Carlo learning
  - TD learning
  - Network architecture có thể tùy chỉnh

- `train.py`: Script huấn luyện agent với các tính năng:
  - Progress bar theo dõi tiến trình
  - Thống kê training theo thời gian thực
  - Lưu model định kỳ và tự động
  - Điều khiển training (pause/resume/stop)

- `play.py`: Script để chạy và đánh giá agent:
  - Tự động load model phù hợp
  - Hiển thị thống kê hiệu suất
  - Đánh giá qua nhiều episodes

## Cấu hình

File `config.json` chứa toàn bộ cấu hình cho:

### Môi trường (env_config)
- Số làn đường và kích thước cửa sổ
- Tốc độ game và FPS
- Hệ thống reward/penalty
- Thông số xe và đường

### Agent (agent_config)
- Tham số học tập (learning rate, gamma)
- Cấu trúc mạng neural
- Kích thước replay buffer
- Device (CPU/GPU)

### Training (training_config)
- Phương pháp học (DQN/Monte Carlo/TD)
- Thời gian hoặc số bước training
- Tần suất cập nhật và lưu model
- Render settings

### Play (play_config)
- Phương pháp học để load model
- Số episode đánh giá

## Huấn luyện Agent

1. Điều chỉnh cấu hình trong `config.json`
2. Chạy script huấn luyện:
```bash
python train.py
```

Trong quá trình training:
- Q: Dừng training và lưu model
- R: Bật/tắt render
- Progress bar hiển thị:
  - Số episodes
  - Score hiện tại
  - Số bước trung bình
  - Thời gian training

## Chạy và Đánh giá Agent

1. Chạy script đánh giá:
```bash
python play.py
```

Script sẽ:
- Load model tương ứng với phương pháp học
- Chạy số episode được cấu hình
- Hiển thị:
  - Thông tin model
  - Điểm trung bình
  - Thời gian sống trung bình

## Cấu trúc Models

```
models/
  ├── dqn_car_dodging/
  │   └── final_model.zip
  ├── monte_carlo_car_dodging/
  │   └── final_model.zip
  └── td_car_dodging/
      └── final_model.zip
```

## Yêu cầu về hình ảnh

Cần có các file hình ảnh sau trong thư mục `images/`:

```
images/
  ├── agents/
  │   └── agent.png
  └── obstacles/
      ├── car1.png
      ├── car2.png
      ...
      └── car12.png
```

## Lưu ý

- Đảm bảo đủ hình ảnh trong thư mục `images/`
- Tạo sẵn thư mục `models/` trước khi training
- Model được tự động lưu theo phương pháp học
- Có thể tiếp tục training từ model đã lưu
- GPU được khuyến nghị cho training (tự phát hiện qua config)
