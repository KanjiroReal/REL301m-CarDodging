# Car Dodging AI

[🇻🇳 Tiếng Việt](README.vi.md)

A simple "Car Dodging" game environment implementation and AI agent training using various reinforcement learning methods (DQN, Monte Carlo, TD Learning).

## Installation

1. Clone repository:
```bash
git clone https://github.com/KanjiroReal/REL301m-CarDodging.git
cd REL301m-CarDodging
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required libraries:
```bash
pip install -r requirements.txt
```

## Project Structure

- `env.py`: CarDodgingEnv environment with features:
  - Configurable multiple lanes
  - Complex reward system (survival, dodge, movement)
  - Detailed information display (score, time, dodge zones)
  - Support for multiple obstacle car types

- `models.py`: Models and learning methods implementation:
  - DQN with replay buffer
  - Monte Carlo learning
  - TD learning
  - Customizable network architecture

- `train.py`: Agent training script with features:
  - Real-time progress tracking
  - Live training statistics
  - Automatic periodic model saving
  - Training controls (pause/resume/stop)

- `play.py`: Agent evaluation script:
  - Automatic model loading
  - Performance statistics display
  - Multi-episode evaluation

## Configuration

`config.json` contains all configurations for:

### Environment (env_config)
- Number of lanes and window size
- Game speed and FPS
- Reward/penalty system
- Car and road parameters

### Agent (agent_config)
- Learning parameters (learning rate, gamma)
- Neural network architecture
- Replay buffer size
- Device (CPU/GPU)

### Training (training_config)
- Learning method (DQN/Monte Carlo/TD)
- Training duration or steps
- Update and save intervals
- Render settings

### Play (play_config)
- Learning method for model loading
- Number of evaluation episodes

## Training Agent

1. Adjust configuration in `config.json`
2. Run training script:
```bash
python train.py
```

During training:
- Q: Stop training and save model
- R: Toggle rendering
- Progress bar shows:
  - Episode count
  - Current score
  - Average steps
  - Training time

## Running and Evaluating Agent

1. Run evaluation script:
```bash
python play.py
```

Script will:
- Load model for selected learning method
- Run configured number of episodes
- Display:
  - Model information
  - Average score
  - Average survival time

## Models Structure

```
models/
  ├── dqn_car_dodging/
  │   └── final_model.zip
  ├── monte_carlo_car_dodging/
  │   └── final_model.zip
  └── td_car_dodging/
      └── final_model.zip
```

## Image Requirements

Required image files in `images/` directory:

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

## Notes

- Ensure all required images are in `images/` directory
- Create `models/` directory before training
- Models are automatically saved based on learning method
- Training can be continued from saved models
- GPU recommended for training (auto-detected via config)
