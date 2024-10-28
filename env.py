import gymnasium as gym
import pygame
import numpy as np
import time

from gymnasium import spaces

# Hằng số
NUM_LANES = 5
AGENT_LIMIT_TICK = 0.3
PENALTY = -50
REWARD = 1
SPAWN_INTERVAL = 1
MAX_STEPS_PER_EPISODE = 10000
OBSTACLE_SPEED = 8

class CarDodgingEnv(gym.Env):
    """Môi trường huấn luyện agent tránh các obstacle trên đường đua nhiều làn
    
    Args:
        Không có tham số đầu vào
        
    Returns:
        CarDodgingEnv: Một instance của môi trường
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        """Khởi tạo môi trường với các thông số cơ bản
        
        Args:
            Không có tham số đầu vào
            
        Returns:
            None
        """
        super().__init__()
        self.window_size = (100 * NUM_LANES, 800)
        self.car_size = (40, 60)
        self.lane_width = self.window_size[0] // NUM_LANES
        
        self.action_space = spaces.Discrete(3)  # 0: left, 1: stay, 2: right
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.Font(None, 24)
        
        self.player_lane = None
        self.obstacles = [None] * NUM_LANES
        self.last_action_time = 0
        self.last_spawn_time = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        """Khởi tạo lại trạng thái môi trường về ban đầu
        
        Args:
            seed (int, optional): Seed cho random state. Default: None
            options (dict, optional): Các tùy chọn bổ sung. Default: None
            
        Returns:
            tuple: (observation, info)
                - observation (np.ndarray): Trạng thái quan sát được
                - info (dict): Thông tin bổ sung về môi trường
        """
        super().reset(seed=seed)
        
        self.player_lane = NUM_LANES // 2
        self.player_pos = [self.lane_width * self.player_lane + (self.lane_width - self.car_size[0]) // 2, 
                           self.window_size[1] - self.car_size[1] - 10]
        self.obstacles = [None] * NUM_LANES
        self.last_action_time = time.time()
        self.last_spawn_time = time.time()
        self.steps = 0
        
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def _spawn_new_obstacle(self):
        """Sinh ra obstacle mới trên một làn đường ngẫu nhiên đang trống
        
        Args:
            Không có tham số đầu vào
            
        Returns:
            None
        """
        empty_lanes = [i for i, obstacle in enumerate(self.obstacles) if obstacle is None]
        if empty_lanes:
            lane = np.random.choice(empty_lanes)
            self.obstacles[lane] = [self.lane_width * lane + (self.lane_width - self.car_size[0]) // 2, 0]

    def step(self, action):
        """Thực hiện một bước trong môi trường
        
        Args:
            action (int): Hành động của agent (0: left, 1: stay, 2: right)
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
                - observation (np.ndarray): Trạng thái quan sát được
                - reward (float): Phần thưởng cho hành động
                - done (bool): Trạng thái kết thúc episode
                - truncated (bool): Luôn là False trong môi trường này
                - info (dict): Thông tin bổ sung về trạng thái môi trường
        """
        self.steps += 1
        current_time = time.time()
        
        if current_time - self.last_action_time >= AGENT_LIMIT_TICK:
            if action == 0 and self.player_lane > 0:
                self.player_lane -= 1
                self.last_action_time = current_time

            elif action == 1:
                self.last_action_time = current_time

            elif action == 2 and self.player_lane < NUM_LANES - 1:
                self.player_lane += 1
                self.last_action_time = current_time

        self.player_pos[0] = self.lane_width * self.player_lane + (self.lane_width - self.car_size[0]) // 2
        
        if current_time - self.last_spawn_time >= SPAWN_INTERVAL:
            self._spawn_new_obstacle()
            self.last_spawn_time = current_time
        
        for i, obstacle in enumerate(self.obstacles):
            if obstacle is not None:
                obstacle[1] += OBSTACLE_SPEED
                if obstacle[1] > self.window_size[1]:
                    self.obstacles[i] = None
        
        # check collision
        done = False
        for obstacle in self.obstacles:
            if obstacle is not None:
                # Kiểm tra va chạm theo hình chữ nhật
                player_left = self.player_pos[0]
                player_right = self.player_pos[0] + self.car_size[0]
                player_top = self.player_pos[1]
                player_bottom = self.player_pos[1] + self.car_size[1]
                
                obstacle_left = obstacle[0]
                obstacle_right = obstacle[0] + self.car_size[0]
                obstacle_top = obstacle[1]
                obstacle_bottom = obstacle[1] + self.car_size[1]
                
                if (player_left < obstacle_right and
                    player_right > obstacle_left and
                    player_top < obstacle_bottom and
                    player_bottom > obstacle_top):
                    done = True
                    break
        
        if done:
            print("Kết thúc epdisode do va chạm!")
            self.render()  # Render lại để hiển thị trạng thái va chạm
            pygame.time.wait(1000)  # Tạm dừng 1 giây
        
        if self.steps >= MAX_STEPS_PER_EPISODE:
            done = True
            print(f"Kết thúc epdisode do đã hết {MAX_STEPS_PER_EPISODE} step!")
        
        reward = REWARD if not done else PENALTY
        observation = self._get_obs()
        info = self._get_state_info()
        
        return observation, reward, done, False, info

    def _get_state_info(self):
        """Lấy thông tin về trạng thái hiện tại của môi trường
        
        Args:
            Không có tham số đầu vào
            
        Returns:
            dict: Dictionary chứa thông tin về:
                - distances: Khoảng cách từ agent đến obstacle trên mỗi làn
                - lane_states: Trạng thái có obstacle hay không trên mỗi làn (0/1)
        """
        distances = []
        for lane in range(NUM_LANES):
            if self.obstacles[lane] is not None:
                distance = self.obstacles[lane][1] - self.player_pos[1]
                distances.append(abs(distance) if distance < 0 else distance)
            else:
                distances.append(float('inf'))
        
        lane_states = [1 if obs is not None else 0 for obs in self.obstacles]
        
        return {
            'distances': distances,
            'lane_states': lane_states
        }

    def render(self):
        """Hiển thị trạng thái hiện tại của môi trường
        
        Args:
            Không có tham số đầu vào
            
        Returns:
            bool: True nếu render thành công, False nếu có lỗi
        """
        try:
            self.screen.fill((255, 255, 255))
            
            # Vẽ làn đường
            for i in range(1, NUM_LANES):
                pygame.draw.line(self.screen, (200, 200, 200), 
                               (i * self.lane_width, 0), 
                               (i * self.lane_width, self.window_size[1]))
            
            # Vẽ agent
            pygame.draw.rect(self.screen, (0, 0, 255), (*self.player_pos, *self.car_size))
            
            # Vẽ obstacles và thông tin khoảng cách
            for i, obstacle in enumerate(self.obstacles):
                if obstacle is not None:
                    pygame.draw.rect(self.screen, (255, 0, 0), (*obstacle, *self.car_size))
                    
                    distance = abs(obstacle[1] - self.player_pos[1])
                    dist_text = self.font.render(f"{int(distance)}", True, (0, 0, 0))
                    self.screen.blit(dist_text, (obstacle[0] + self.car_size[0]/2 - 15, obstacle[1] + self.car_size[1]))
            
            # Hiển thị trạng thái làn đường
            lane_states = self._get_state_info()['lane_states']
            for i, state in enumerate(lane_states):
                state_text = self.font.render(f"Lane {i}: {'1' if state == 1 else '0'}", True, (0, 0, 0))
                self.screen.blit(state_text, (10 + i * self.lane_width, 10))
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return True
            
        except Exception as e:
            print(f"Lỗi khi render: \n{str(e)}")
            return False

    def _get_obs(self):
        """Lấy observation từ trạng thái hiện tại của môi trường
        
        Args:
            Không có tham số đầu vào
            
        Returns:
            np.ndarray: Ma trận RGB thể hiện trạng thái màn hình
        """
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        """Đóng môi trường và giải phóng tài nguyên
        
        Args:
            Không có tham số đầu vào
            
        Returns:
            None
        """
        pygame.quit()


def main(log: bool = False) -> None:
    """Hàm chạy thử nghiệm môi trường với các hành động ngẫu nhiên
    
    Args:
        log (bool): có hiển thị khoảng cách cho các chướng ngại vật tới xe hay không. Default: False
        
    Returns:
        None
    """
    env = CarDodgingEnv()
    num_episodes = 2

    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Xử lý sự kiện pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        env.close()
                        return
            
            action = env.action_space.sample()
            observation, reward, done, _, info = env.step(action)
            
            if log:
                print(f"Episode {episode + 1}")
                print(f"Distances to obstacles: {info['distances']}")
                print(f"Lane states: {info['lane_states']}")
                print("---")
            
            env.render()
            total_reward += reward

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
    
    env.close()


if __name__ == "__main__":
    main()