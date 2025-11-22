import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque
import heapq

class MouseMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, maze_data, optimal_path=None, render_mode=None):
        super(MouseMazeEnv, self).__init__()

        self.maze = maze_data
        self.height, self.width = self.maze.shape
        self.render_mode = render_mode

        # Actions: 0-3 (Walk: U, R, D, L), 4-7 (Run: U, R, D, L)
        self.action_space = spaces.Discrete(8)

        # Observation contains egocentric view, orientation one-hot, turn memory,
        # current velocity, and smell intensity toward the cheese.
        # 3 (front/left/right view) + 4 (orientation) + 3 (turn memory) + 1 (velocity) + 1 (smell)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Find start and goal
        self.start_pos = tuple(np.argwhere(self.maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 3)[0])
        self.agent_pos = self.start_pos

        self.window = None
        self.clock = None
        self.cell_size = 20

        # Clean maze for internal state (remove start marker so it doesn't persist)
        self.grid = self.maze.copy()
        self.grid[self.start_pos] = 0

        # Orientation is 0=Up,1=Right,2=Down,3=Left. Start facing right.
        self.orientation = 1
        self.turn_history = deque(maxlen=3)
        for _ in range(3):
            self.turn_history.append(0)

        # Velocity expresses how quickly the mouse is trying to move (1=walk, 2=run).
        self.velocity = 1.0

        # Optimal path and distance field enable olfactory shaping and deviation penalties.
        self.optimal_path, _ = self._compute_optimal_path() if optimal_path is None else ( [tuple(p) for p in optimal_path], len(optimal_path) - 1 )
        self.optimal_index_map = {pos: idx for idx, pos in enumerate(self.optimal_path)}
        self.last_optimal_index = 0
        self.distance_field = self._build_distance_field()

        self.max_steps = self.height * self.width * 4
        self.step_count = 0
        self.max_stall = max(10, (self.height + self.width))
        self.stall_steps = 0
        self.last_smell = self._smell_intensity()
        self.visit_counts = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.orientation = 1
        self.turn_history.clear()
        for _ in range(3):
            self.turn_history.append(0)
        self.velocity = 1.0
        self.last_optimal_index = 0
        self.step_count = 0
        self.stall_steps = 0
        self.last_smell = self._smell_intensity()
        self.visit_counts = {}
        return self._get_obs(), {}

    def _get_obs(self):
        visible = self._get_visible_cells()
        orientation = np.eye(4)[self.orientation].astype(np.float32)
        turn_memory = np.array([self._encode_turn(t) for t in self.turn_history], dtype=np.float32)
        smell = np.array([self._smell_intensity()], dtype=np.float32)
        speed = np.array([self.velocity / 2.0], dtype=np.float32)  # normalize to [0,1]
        obs = np.concatenate([visible, orientation, turn_memory, speed, smell]).astype(np.float32)
        return obs

    def step(self, action):
        # Directions: Up, Right, Down, Left
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        is_running = action >= 4
        direction_idx = action % 4
        dr, dc = deltas[direction_idx]

        # Track turn memory (relative to prior orientation)
        prev_orientation = self.orientation
        self._record_turn(direction_idx)
        self.orientation = direction_idx

        # Velocity dynamics: accelerate on consecutive runs, slow when walking or turning.
        target_speed = 2.0 if is_running else 1.0
        if is_running and direction_idx == prev_orientation:
            self.velocity = min(2.0, self.velocity + 0.25)
        else:
            self.velocity = max(1.0, self.velocity - 0.25)
        self.velocity = 0.7 * self.velocity + 0.3 * target_speed

        reward = 0.0
        terminated = False
        truncated = False

        r, c = self.agent_pos

        prev_dist = self.distance_field[r, c]
        steps_to_take = 2 if self.velocity >= 1.5 or is_running else 1
        crash = False

        for _ in range(steps_to_take):
            nr, nc = r + dr, c + dc

            # Check Bounds and Walls
            if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] != 1:
                r, c = nr, nc
                if (r, c) == self.goal_pos:
                    reward += 100.0
                    terminated = True
                    break
            else:
                crash = True
                break

        if crash:
            reward -= 10.0 if is_running else 5.0
            self.velocity = max(1.0, self.velocity - 0.5)
        else:
            self.agent_pos = (r, c)

            # Reward for keeping momentum through straight hallways
            if is_running and self._is_straight_corridor(self.agent_pos, direction_idx):
                reward += 0.3
            # Penalty for speeding into corners
            if is_running and self._approaching_corner(self.agent_pos, direction_idx):
                reward -= 1.0

        # Olfactory shaping: only benefits when an open-path distance shrinks
        new_dist = self.distance_field[self.agent_pos]
        if np.isfinite(prev_dist) and np.isfinite(new_dist):
            reward += 0.5 * (prev_dist - new_dist)

        # Stagnation handling: detect lack of smell progress and repeated visits
        smell_now = self._smell_intensity()
        if smell_now > self.last_smell + 1e-3:
            self.last_smell = smell_now
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        self.visit_counts[self.agent_pos] = self.visit_counts.get(self.agent_pos, 0) + 1
        if self.visit_counts[self.agent_pos] > 3:
            reward -= 0.05 * (self.visit_counts[self.agent_pos] - 3)

        # Deviations from optimal route get penalized; progress is free
        if not terminated and not crash:
            opt_idx = self.optimal_index_map.get(self.agent_pos, None)
            if opt_idx is not None and opt_idx >= self.last_optimal_index:
                self.last_optimal_index = opt_idx
            else:
                deviation = self._distance_to_optimal(self.agent_pos)
                reward -= 0.2 * (1 + deviation)

        self.step_count += 1
        if self.step_count >= self.max_steps or self.stall_steps > self.max_stall:
            truncated = True
            reward -= 2.0

        info = {"is_success": terminated}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))
        canvas.fill((255, 255, 255))
        
        for r in range(self.height):
            for c in range(self.width):
                color = (255, 255, 255) # Path
                if self.grid[r, c] == 1:
                    color = (0, 0, 0) # Wall
                elif (r, c) == self.goal_pos:
                    color = (255, 215, 0) # Cheese (Gold)
                
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size
                    ),
                )
        
        # Draw Agent (Mouse)
        agent_color = (128, 128, 128) # Grey
        pygame.draw.circle(
            canvas,
            agent_color,
            (int((self.agent_pos[1] + 0.5) * self.cell_size), int((self.agent_pos[0] + 0.5) * self.cell_size)),
            self.cell_size // 3,
        )

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # --- Helper methods ---
    def _record_turn(self, new_orientation):
        delta = (new_orientation - self.orientation) % 4
        if delta == 0:
            turn = 0            # straight
        elif delta == 1:
            turn = 1            # right
        elif delta == 3:
            turn = -1           # left
        else:
            turn = 2            # about-face
        self.turn_history.append(turn)

    def _encode_turn(self, turn):
        # Map turn values to [0,1] for observation stability
        if turn == -1:
            return 0.0
        if turn == 0:
            return 0.33
        if turn == 1:
            return 0.66
        return 1.0

    def _get_visible_cells(self):
        # Return egocentric observations for front, left, and right cells.
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        offsets = [0, -1, 1]  # front, left, right relative to orientation
        visible = []
        for off in offsets:
            dir_idx = (self.orientation + off) % 4
            dr, dc = directions[dir_idx]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                visible.append(1.0)  # wall/out-of-bounds blocks vision
            elif (nr, nc) == self.goal_pos:
                visible.append(0.5)  # cheese ahead smells stronger
            elif self.grid[nr, nc] == 1:
                visible.append(1.0)
            else:
                visible.append(0.0)
        return np.array(visible, dtype=np.float32)

    def _build_distance_field(self):
        # BFS from goal to compute open-path distances (used as smell signal).
        dist = np.full_like(self.grid, np.inf, dtype=np.float32)
        q = [self.goal_pos]
        dist[self.goal_pos] = 0.0
        head = 0
        while head < len(q):
            r, c = q[head]
            head += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] != 1:
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))
        return dist

    def _smell_intensity(self):
        d = self.distance_field[self.agent_pos]
        if not np.isfinite(d):
            return 0.0
        return 1.0 / (1.0 + d)

    def _compute_optimal_path(self):
        start = self.start_pos
        goal = self.goal_pos
        queue = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        while queue:
            _, current = heapq.heappop(queue)
            if current == goal:
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] != 1:
                    new_cost = cost_so_far[current] + 1
                    next_node = (nr, nc)
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + abs(goal[0] - nr) + abs(goal[1] - nc)
                        heapq.heappush(queue, (priority, next_node))
                        came_from[next_node] = current
        if goal not in came_from:
            return [start], 0
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.append(start)
        path.reverse()
        return path, len(path) - 1

    def _distance_to_optimal(self, pos):
        # Manhattan distance to the closest node on the optimal path
        best = min(abs(pos[0] - r) + abs(pos[1] - c) for r, c in self.optimal_path)
        return best

    def _is_straight_corridor(self, pos, direction_idx):
        # Reward safe acceleration on long straight segments.
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = directions[direction_idx]
        ahead1 = (pos[0] + dr, pos[1] + dc)
        ahead2 = (pos[0] + 2 * dr, pos[1] + 2 * dc)
        def is_open(p):
            r, c = p
            return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1
        left_idx = (direction_idx - 1) % 4
        right_idx = (direction_idx + 1) % 4
        left_open = is_open((pos[0] + directions[left_idx][0], pos[1] + directions[left_idx][1]))
        right_open = is_open((pos[0] + directions[right_idx][0], pos[1] + directions[right_idx][1]))
        return is_open(ahead1) and is_open(ahead2) and not left_open and not right_open

    def _approaching_corner(self, pos, direction_idx):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = directions[direction_idx]
        ahead = (pos[0] + dr, pos[1] + dc)
        def is_open(p):
            r, c = p
            return 0 <= r < self.height and 0 <= c < self.width and self.grid[r, c] != 1
        if not is_open(ahead):
            return True
        left_idx = (direction_idx - 1) % 4
        right_idx = (direction_idx + 1) % 4
        left_open = is_open((pos[0] + directions[left_idx][0], pos[1] + directions[left_idx][1]))
        right_open = is_open((pos[0] + directions[right_idx][0], pos[1] + directions[right_idx][1]))
        return left_open or right_open
