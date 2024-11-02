import gymnasium as gym
import pygame
import numpy as np
import time

from gymnasium import spaces

class CarDodgingEnv(gym.Env):
    """Môi trường huấn luyện agent tránh các obstacle trên đường đua nhiều làn
    
    Args:
        config (dict, optional): Cấu hình cho môi trường. Default: None
        
    Returns:
        CarDodgingEnv: Một instance của môi trường
    """
    
    metadata = {"render_modes": ["human"], "render_fps": None}

    def __init__(self, config=None):
        """Khởi tạo môi trường với config tùy chọn
        
        Args:
            config (dict, optional): Cấu hình cho môi trường. Default: None
        """
        super().__init__()
        
        # Load config hoặc sử dụng giá trị mặc định
        if config is None:
            config = {
                "num_lanes": 5,
                "agent_limit_tick": 0.3,
                "penalty": -50,
                "reward": 1,
                "spawn_interval": 0.5,
                "max_steps": 10000,
                "obstacle_speed": 480,
                "render_fps": 120,
                "window_size": {
                    "game_width": 500,
                    "game_height": 800,
                    "info_panel_height": 50,
                    "side_panel_width": 200
                },
                "car_size": {
                    "width": 40,
                    "height": 60
                },
                "colors": {
                    "road": [64, 64, 64],
                    "line": [255, 255, 0]
                }
            }
        
        # Khởi tạo các thông số từ config
        self.num_lanes = config["num_lanes"]
        self.agent_limit_tick = config["agent_limit_tick"]
        self.spawn_interval = config["spawn_interval"]
        self.max_steps = config["max_steps"]
        self.obstacle_speed = config["obstacle_speed"]
        
        self.metadata = {"render_modes": ["human"], "render_fps": config["render_fps"]}
        
        # Kích thước cửa sổ và panel
        self.info_panel_height = config["window_size"]["info_panel_height"]
        self.side_panel_width = config["window_size"]["side_panel_width"]
        self.game_window_size = (
            config["window_size"]["game_width"],
            config["window_size"]["game_height"]
        )
        
        self.window_size = (
            self.game_window_size[0] + self.side_panel_width,
            self.game_window_size[1] + self.info_panel_height
        )
        
        self.car_size = (
            config["car_size"]["width"],
            config["car_size"]["height"]
        )
        
        self.lane_width = self.game_window_size[0] // self.num_lanes
        
        # Màu sắc
        self.road_color = tuple(config["colors"]["road"])
        self.line_color = tuple(config["colors"]["line"])
        
        self.action_space = spaces.Discrete(3)
        # Điều chỉnh observation space để phù hợp với kích thước mới
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.Font(None, 24)
        
        self.player_lane = None
        self.obstacles = [None] * self.num_lanes
        self.last_action_time = 0
        self.last_spawn_time = 0
        self.steps = 0
        self.last_update_time = time.time()
        self.fixed_timestep = 1.0 / 60  # 60Hz game logic
        self.max_reward = 0  # Thêm biến theo dõi max reward
        self.current_reward = 0  # Thêm biến theo dõi reward hiện tại
        
        # Load images
        self.agent_img = pygame.image.load('images/agents/agent.png')
        self.agent_img = pygame.transform.scale(self.agent_img, self.car_size)
        
        # Load obstacle images
        self.obstacle_imgs = []
        for i in range(1, 13):  # Load car1.png to car12.png
            img = pygame.image.load(f'images/obstacles/car{i}.png')
            img = pygame.transform.scale(img, self.car_size)
            self.obstacle_imgs.append(img)
            
        # Dictionary để lưu trữ obstacle image được chọn cho mỗi obstacle
        self.obstacle_images = {}
        
        # Load reward config
        reward_config = config.get("reward_config", {
            "reward_interval": 1.0,
            "points_per_second": 1,
            "collision_penalty": -10
        })
        
        self.reward_interval = reward_config["reward_interval"]
        self.points_per_second = reward_config["points_per_second"]
        self.collision_penalty = reward_config["collision_penalty"]
        
        # Thêm biến theo dõi thời gian
        self.start_time = None
        self.elapsed_time = 0
        self.last_reward_time = 0
        self.current_reward = 0
        self.max_reward = 0
        
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
        
        self.player_lane = self.num_lanes // 2
        self.player_pos = [self.lane_width * self.player_lane + (self.lane_width - self.car_size[0]) // 2, 
                           self.game_window_size[1] - self.car_size[1] - 10]
        self.obstacles = [None] * self.num_lanes
        self.last_action_time = time.time()
        self.last_spawn_time = time.time()
        self.steps = 0
        self.current_reward = 0  # Reset reward hiện tại
        
        self.obstacle_images = {}  # Reset obstacle images dictionary
        
        # Reset các biến thời gian
        self.start_time = time.time()
        self.last_reward_time = self.start_time
        self.elapsed_time = 0
        self.current_reward = 0
        
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def _spawn_new_obstacle(self):
        """Sinh ra obstacle mới trên một làn đường ngẫu nhiên đang trống
        
        Args:
            Không c tham số đầu vào
            
        Returns:
            None
        """
        empty_lanes = [i for i, obstacle in enumerate(self.obstacles) if obstacle is None]
        if empty_lanes:
            lane = np.random.choice(empty_lanes)
            self.obstacles[lane] = [self.lane_width * lane + (self.lane_width - self.car_size[0]) // 2, 0]
            # Chọn ngẫu nhiên một hình ảnh obstacle cho obstacle mới
            self.obstacle_images[lane] = np.random.choice(self.obstacle_imgs)

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
        elapsed = current_time - self.last_update_time
        
        # Cập nhật thời gian trôi qua
        self.elapsed_time = current_time - self.start_time
        
        # Tính số giây đã trôi qua kể từ lần reward cuối
        time_since_last_reward = current_time - self.last_reward_time
        
        # Tính số reward cần cộng thêm
        reward_points = int(time_since_last_reward / self.reward_interval) * self.points_per_second
        
        if reward_points > 0:
            # Cập nhật thời điểm reward cuối
            self.last_reward_time = current_time - (time_since_last_reward % self.reward_interval)
        
        # Update game logic với timestep cố định
        self.last_update_time = current_time
        
        # Cập nhật vị trí obstacle dựa trên thời gian thực
        for i, obstacle in enumerate(self.obstacles):
            if obstacle is not None:
                obstacle[1] += self.obstacle_speed * elapsed
                if obstacle[1] > self.game_window_size[1]:
                    self.obstacles[i] = None
        
        # Spawn obstacle dựa trên thời gian thực
        if current_time - self.last_spawn_time >= self.spawn_interval:
            self._spawn_new_obstacle()
            self.last_spawn_time = current_time
        
        if current_time - self.last_action_time >= self.agent_limit_tick:
            if action == 0 and self.player_lane > 0:
                self.player_lane -= 1
                self.last_action_time = current_time

            elif action == 1:
                self.last_action_time = current_time

            elif action == 2 and self.player_lane < self.num_lanes - 1:
                self.player_lane += 1
                self.last_action_time = current_time

        self.player_pos[0] = self.lane_width * self.player_lane + (self.lane_width - self.car_size[0]) // 2
        
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
            # print("\nKết thúc epdisode do va chạm!")
            self.render()  # Render lại để hiển thị trạng thái va chạm
            pygame.time.wait(1000)  # Tạm dừng 1 giây
        
        if self.steps >= self.max_steps:
            done = True
            # print(f"\nKết thúc epdisode do đã hết {self.max_steps} step!")
        
        if done:
            step_reward = self.collision_penalty
        else:
            # Tính reward dựa trên thời gian sống sót
            step_reward = reward_points
        
        # Cập nhật current_reward và max_reward
        self.current_reward += step_reward
        self.max_reward = max(self.max_reward, self.current_reward)
        
        if self.steps >= self.max_steps:
            done = True
        
        observation = self._get_obs()
        info = self._get_state_info()
        
        # Thêm thông tin thời gian và điểm vào info
        info.update({
            'elapsed_time': self.elapsed_time,
            'current_reward': self.current_reward,
            'points_per_second': self.points_per_second
        })
        
        return observation, step_reward, done, False, info

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
        for lane in range(self.num_lanes):
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
            
            # 1. Vẽ panel thông tin phía trên
            top_panel = pygame.Surface((self.game_window_size[0], self.info_panel_height))
            top_panel.fill((240, 240, 240))
            
            # Vẽ trng thái làn đường trên panel trên
            for i in range(self.num_lanes):
                has_obstacle = self.obstacles[i] is not None
                color = (255, 0, 0) if has_obstacle else (0, 150, 0)
                status = 1 if has_obstacle else 0
                
                lane_text = self.font.render(f"Lane {i + 1}: {status}", True, color)
                text_x = i * self.lane_width + (self.lane_width - lane_text.get_width()) // 2
                top_panel.blit(lane_text, (text_x, 15))
            
            # Vẽ đường phân cách dưới panel trên
            pygame.draw.line(top_panel, (150, 150, 150), 
                           (0, self.info_panel_height - 1),
                           (self.game_window_size[0], self.info_panel_height - 1), 2)
            
            # 2. Vẽ panel bên phải
            side_panel = pygame.Surface((self.side_panel_width, self.window_size[1]))
            side_panel.fill((240, 240, 240))
            
            # Hiển thị thời gian
            time_text = self.font.render(f"Time: {self.elapsed_time:.1f}s", True, (0, 0, 0))
            
            # Hiển thị điểm
            current_reward_text = self.font.render(f"Score: {self.current_reward}", True, (0, 100, 0))
            max_reward_text = self.font.render(f"Best: {self.max_reward}", True, (0, 0, 100))
            
            side_panel.blit(time_text, (10, 20))
            side_panel.blit(current_reward_text, (10, 50))
            side_panel.blit(max_reward_text, (10, 80))
            
            side_panel.blit(self.font.render(f"Survive in {self.reward_interval:.0f}s: +{self.points_per_second}", True, (0, 200, 0)), (10, 110))
            side_panel.blit(self.font.render(f"Collision: {self.collision_penalty}", True, (255, 0, 0)), (10, 140))
            
            # Vẽ đường phân cách bên trái panel phải
            pygame.draw.line(self.screen, (150, 150, 150),
                           (self.game_window_size[0], 0),
                           (self.game_window_size[0], self.window_size[1]), 2)
            
            # 3. Vẽ phần game
            game_surface = pygame.Surface(self.game_window_size)
            
            # Vẽ nền đường màu xám
            game_surface.fill(self.road_color)
            
            # Vẽ vạch kẻ đường màu vàng nét đứt
            for i in range(1, self.num_lanes):
                x = i * self.lane_width
                # Vẽ các đoạn nét đứt
                dash_length = 30  # Độ dài mỗi nét
                gap_length = 20   # Khoảng cách giữa các nét
                y = 0
                while y < self.game_window_size[1]:
                    pygame.draw.line(game_surface, self.line_color,
                                   (x, y),
                                   (x, min(y + dash_length, self.game_window_size[1])), 2)
                    y += dash_length + gap_length
            
            # Vẽ agent bằng hình ảnh
            game_surface.blit(self.agent_img, self.player_pos)
            
            # Vẽ obstacles bằng hình ảnh được chọn ngẫu nhiên
            for i, obstacle in enumerate(self.obstacles):
                if obstacle is not None:
                    # Lấy hình ảnh đã được chọn cho obstacle này
                    obstacle_img = self.obstacle_images.get(i)
                    if obstacle_img is None:
                        # Nếu chưa có hình ảnh, chọn một hình ngẫu nhiên
                        obstacle_img = np.random.choice(self.obstacle_imgs)
                        self.obstacle_images[i] = obstacle_img
                    
                    game_surface.blit(obstacle_img, obstacle)
                    
                    # Vẽ khoảng cách
                    distance = abs(obstacle[1] - self.player_pos[1])
                    dist_text = self.font.render(f"{int(distance)}", True, (255, 255, 255))  # Đổi màu text thành trắng
                    game_surface.blit(dist_text, (obstacle[0] + self.car_size[0] + 5, obstacle[1]))
            
            # 4. Vẽ tất cả lên main screen
            self.screen.blit(top_panel, (0, 0))  # Panel trên
            self.screen.blit(side_panel, (self.game_window_size[0], 0))  # Panel phải
            self.screen.blit(game_surface, (0, self.info_panel_height))  # Phần game
            
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
    num_episodes = 100

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