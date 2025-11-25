import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from collections import deque
import heapq


def compute_sensory_inputs(grid, x, y, direction, last_action, last_last_action, goal_pos):
    """
    Computes the sensory observation for the mouse agent using a FoV cone.
    grid values: 0=free, 1=wall, 2=goal.
    """
    DIR = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
    LEFT = {"N": (-1, 0), "E": (0, -1), "S": (1, 0), "W": (0, 1)}
    RIGHT = {"N": (1, 0), "E": (0, 1), "S": (-1, 0), "W": (0, -1)}

    def get_tile(tx, ty):
        if 0 <= tx < grid.shape[1] and 0 <= ty < grid.shape[0]:
            return grid[ty, tx]
        return 1  # outside bounds is treated as wall

    dx, dy = DIR[direction]
    lx, ly = LEFT[direction]
    rx, ry = RIGHT[direction]

    # Immediate surroundings
    wall_left = get_tile(x + lx, y + ly)
    wall_front = get_tile(x + dx, y + dy)
    wall_right = get_tile(x + rx, y + ry)
    cl = wall_left
    cr = wall_right

    # Forward tiles
    a1 = get_tile(x + dx, y + dy)
    a2 = get_tile(x + 2 * dx, y + 2 * dy)
    a1_l = get_tile(x + dx + lx, y + dy + ly)
    a1_r = get_tile(x + dx + rx, y + dy + ry)
    a2_l = get_tile(x + 2 * dx + lx, y + 2 * dy + ly)
    a2_r = get_tile(x + 2 * dx + rx, y + 2 * dy + ry)

    # Smell intensity = normalized inverse distance to goal (observation only)
    gx, gy = goal_pos
    dist = np.linalg.norm([x - gx, y - gy])
    smell_intensity = np.clip(1.0 / (dist + 1e-6), 0, 1)

    obs = np.array(
        [
            wall_left,
            wall_front,
            wall_right,
            cl,
            cr,
            a1,
            a1_l,
            a1_r,
            a2,
            a2_l,
            a2_r,
            smell_intensity,
            *last_action,
            *last_last_action,
        ],
        dtype=np.float32,
    )
    return obs


class MouseMazeEnv(gym.Env):
    """
    Turn-and-move FoV maze (grid provided by caller):
    - Actions: 0=forward, 1=turn left, 2=turn right, 3=turn back
    - Observation: 3 whisker + 2 lateral + 6 forward-cone + smell + last_action(4) + last_last(4)
    - Rewards: step cost, crash penalty, inverse-distance shaping, goal reward
    - No smell or optimal-path penalties in reward; smell is observation only
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, maze_data, optimal_path=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.grid = maze_data.copy()
        self.height, self.width = self.grid.shape
        self.grid_size = self.height  # assume square grids as before

        start_locs = np.argwhere(self.grid == 2)
        goal_locs = np.argwhere((self.grid == 3) | (self.grid == 2))  # accept 3 as goal, fall back to 2
        if len(start_locs) == 0:
            raise ValueError("Start position (value 2) not found in maze_data")
        if len(goal_locs) == 0:
            raise ValueError("Goal position (value 3) not found in maze_data")

        self.start_pos = tuple(start_locs[0][::-1])  # (x, y)
        # Prefer explicit goal marker 3; otherwise reuse start marker if provided that way
        goal_candidates = np.argwhere(self.grid == 3)
        goal = goal_candidates[0] if len(goal_candidates) > 0 else goal_locs[0]
        self.goal_pos = (goal[1], goal[0])  # (x, y)

        self.agent_x, self.agent_y = self.start_pos
        self.grid[self.agent_y, self.agent_x] = 0  # clear start cell

        self.action_space = spaces.Discrete(4)
        obs_len = 3 + 2 + 6 + 1 + 4 + 4
        # Grid can contain values up to 3 (walls/goals); smell is <=1
        self.observation_space = spaces.Box(low=0, high=3, shape=(obs_len,), dtype=np.float32)

        self.direction = "E"
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_last_action = np.zeros(4, dtype=np.float32)

        self.step_penalty = -0.05
        self.crash_penalty = -1.0
        self.goal_reward = 10.0
        self.dist_reward_scale = 0.2

        self.max_steps = max(50, self.height * self.width * 4)
        self.step_count = 0

        self.window = None
        self.clock = None
        self.cell_size = 20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset to initial positions
        self.grid[self.goal_pos[1], self.goal_pos[0]] = 3
        self.agent_x, self.agent_y = self.start_pos
        self.direction = "E"
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_last_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0

        obs = compute_sensory_inputs(
            self.grid,
            self.agent_x,
            self.agent_y,
            self.direction,
            self.last_action,
            self.last_last_action,
            self.goal_pos,
        )
        return obs, {}

    def step(self, action):
        self.last_last_action = self.last_action.copy()
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_action[action] = 1.0

        DIR = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

        if action == 1:
            self.direction = {"N": "W", "W": "S", "S": "E", "E": "N"}[self.direction]
        elif action == 2:
            self.direction = {"N": "E", "E": "S", "S": "W", "W": "N"}[self.direction]
        elif action == 3:
            self.direction = {"N": "S", "S": "N", "E": "W", "W": "E"}[self.direction]

        reward = 0.0
        terminated = False
        truncated = False

        dx, dy = DIR[self.direction]
        nx, ny = self.agent_x + dx, self.agent_y + dy

        crashed = self.grid[ny, nx] == 1
        if crashed:
            reward += self.crash_penalty
        else:
            self.agent_x, self.agent_y = nx, ny
            reward += self.step_penalty

        gx, gy = self.goal_pos
        dist_now = np.linalg.norm([self.agent_x - gx, self.agent_y - gy])
        reward += (1.0 / (dist_now + 1e-6)) * self.dist_reward_scale

        if (self.agent_x, self.agent_y) == self.goal_pos:
            reward += self.goal_reward
            terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        obs = compute_sensory_inputs(
            self.grid,
            self.agent_x,
            self.agent_y,
            self.direction,
            self.last_action,
            self.last_last_action,
            self.goal_pos,
        )
        return obs, reward, terminated, truncated, {"is_success": terminated}

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill((255, 255, 255))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = (255, 255, 255)
                if self.grid[y, x] == 1:
                    color = (0, 0, 0)
                elif self.grid[y, x] == 2:
                    color = (255, 215, 0)
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size
                    ),
                )

        pygame.draw.circle(
            canvas,
            (128, 128, 128),
            (
                int((self.agent_x + 0.5) * self.cell_size),
                int((self.agent_y + 0.5) * self.cell_size),
            ),
            self.cell_size // 3,
        )

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class MouseMazeEnvLegacy(gym.Env):
    """
    Legacy straight-move FoV maze (4 actions Up/Right/Down/Left).
    Observation: 8-tile FoV + orientation one-hot + 3-step turn memory + smell (distance-based).
    Rewards: goal +100, crash -5, smell-based shaping, visit penalty, optional optimal-path deviation penalty.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, maze_data, optimal_path=None, render_mode=None, use_smell=True, smell_in_obs=True, use_optimal_penalty=True):
        super().__init__()
        self.maze = maze_data
        self.height, self.width = self.maze.shape
        self.render_mode = render_mode
        self.use_smell = use_smell
        self.smell_in_obs = smell_in_obs
        self.use_optimal_penalty = use_optimal_penalty

        self.action_space = spaces.Discrete(4)
        obs_dim = 8 + 4 + 3 + (1 if self.smell_in_obs else 0)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.start_pos = tuple(np.argwhere(self.maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 3)[0])
        self.agent_pos = self.start_pos

        self.window = None
        self.clock = None
        self.cell_size = 20

        self.grid = self.maze.copy()
        self.grid[self.start_pos] = 0

        self.orientation = 1  # start facing right
        self.turn_history = deque(maxlen=3)
        for _ in range(3):
            self.turn_history.append(0)

        self.optimal_path, _ = self._compute_optimal_path() if optimal_path is None else ([tuple(p) for p in optimal_path], len(optimal_path) - 1)
        self.optimal_index_map = {pos: idx for idx, pos in enumerate(self.optimal_path)}
        self.last_optimal_index = 0
        self.distance_field = self._build_distance_field()

        self.max_steps = self.height * self.width * 10
        self.step_count = 0
        self.max_stall = max(10, (self.height + self.width))
        self.stall_steps = 0
        self.last_smell = self._smell_intensity() if self.use_smell else None
        self.visit_counts = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.orientation = 1
        self.turn_history.clear()
        for _ in range(3):
            self.turn_history.append(0)
        self.last_optimal_index = 0
        self.step_count = 0
        self.stall_steps = 0
        self.last_smell = self._smell_intensity() if self.use_smell else None
        self.visit_counts = {}
        return self._get_obs(), {}

    def _get_obs(self):
        fov = self._get_fov_tiles()
        orientation = np.eye(4)[self.orientation].astype(np.float32)
        turn_memory = np.array([self._encode_turn(t) for t in self.turn_history], dtype=np.float32)
        smell = np.array([self._smell_intensity()], dtype=np.float32) if self.smell_in_obs else np.array([], dtype=np.float32)
        obs = np.concatenate([fov, orientation, turn_memory, smell]).astype(np.float32)
        return obs

    def step(self, action):
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        direction_idx = action % 4
        dr, dc = deltas[direction_idx]

        prev_orientation = self.orientation
        self._record_turn(direction_idx)
        self.orientation = direction_idx

        reward = 0.0
        terminated = False
        truncated = False

        r, c = self.agent_pos
        prev_dist = self.distance_field[r, c]
        crash = False

        nr, nc = r + dr, c + dc
        if 0 <= nr < self.height and 0 <= nc < self.width and self.grid[nr, nc] != 1:
            r, c = nr, nc
            if (r, c) == self.goal_pos:
                reward += 100.0
                terminated = True
        else:
            crash = True

        if crash:
            reward -= 5.0
        else:
            self.agent_pos = (r, c)

        if self.use_smell:
            new_dist = self.distance_field[self.agent_pos]
            if np.isfinite(prev_dist) and np.isfinite(new_dist):
                reward += 0.5 * (prev_dist - new_dist)

            smell_now = self._smell_intensity()
            if smell_now > self.last_smell + 1e-3:
                self.last_smell = smell_now
                self.stall_steps = 0
            else:
                self.stall_steps += 1
        else:
            if crash:
                self.stall_steps += 1
            else:
                self.stall_steps = 0

        self.visit_counts[self.agent_pos] = self.visit_counts.get(self.agent_pos, 0) + 1
        if self.visit_counts[self.agent_pos] > 3:
            reward -= 0.05 * (self.visit_counts[self.agent_pos] - 3)

        if self.use_optimal_penalty and not terminated and not crash:
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

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"is_success": terminated}

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
                color = (255, 255, 255)
                if self.grid[r, c] == 1:
                    color = (0, 0, 0)
                elif (r, c) == self.goal_pos:
                    color = (255, 215, 0)
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size),
                )

        agent_color = (128, 128, 128)
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

    # Helper methods
    def _record_turn(self, new_orientation):
        delta = (new_orientation - self.orientation) % 4
        if delta == 0:
            turn = 0
        elif delta == 1:
            turn = 1
        elif delta == 3:
            turn = -1
        else:
            turn = 2
        self.turn_history.append(turn)

    def _encode_turn(self, turn):
        if turn == -1:
            return 0.0
        if turn == 0:
            return 0.33
        if turn == 1:
            return 0.66
        return 1.0

    def _get_fov_tiles(self):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dir_vec = directions[self.orientation]
        left_vec = directions[(self.orientation - 1) % 4]
        right_vec = directions[(self.orientation + 1) % 4]

        def sample(pos):
            r, c = pos
            if not (0 <= r < self.height and 0 <= c < self.width):
                return 1.0
            if self.grid[r, c] == 1:
                return 1.0
            return 0.0

        x, y = self.agent_pos
        cl = sample((x + left_vec[0], y + left_vec[1]))
        cr = sample((x + right_vec[0], y + right_vec[1]))

        a1_pos = (x + dir_vec[0], y + dir_vec[1])
        a2_pos = (x + 2 * dir_vec[0], y + 2 * dir_vec[1])

        a1 = sample(a1_pos)
        a1l = sample((a1_pos[0] + left_vec[0], a1_pos[1] + left_vec[1]))
        a1r = sample((a1_pos[0] + right_vec[0], a1_pos[1] + right_vec[1]))

        a2 = sample(a2_pos)
        a2l = sample((a2_pos[0] + left_vec[0], a2_pos[1] + left_vec[1]))
        a2r = sample((a2_pos[0] + right_vec[0], a2_pos[1] + right_vec[1]))

        return np.array([cl, cr, a1, a1l, a1r, a2, a2l, a2r], dtype=np.float32)

    def _build_distance_field(self):
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
        best = min(abs(pos[0] - r) + abs(pos[1] - c) for r, c in self.optimal_path)
        return best

class SimpleMazeEnv(gym.Env):
    """Simplified maze environment designed to be solvable by evolutionary algorithms."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, maze_data, render_mode=None):
        super().__init__()

        self.maze = maze_data
        self.height, self.width = self.maze.shape

        # Actions: Up, Right, Down, Left
        self.action_space = spaces.Discrete(4)

        # Observation:
        # [wall_up, wall_right, wall_down, wall_left, goal_dx_norm, goal_dy_norm]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.render_mode = render_mode

        self.start_pos = tuple(np.argwhere(self.maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 3)[0])
        self.agent_pos = self.start_pos

        self.max_steps = max(10, self.height * self.width)
        self.steps = 0

        self.window = None
        self.clock = None
        self.cell_size = 20

        # Precompute greedy guidance toward goal (shortest-path actions).
        self._best_action_map, self._dist_field = self._build_best_action_map()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        r, c = self.agent_pos

        # Wall sensors
        wall_up = 1.0 if (r - 1 < 0 or self.maze[r - 1, c] == 1) else 0.0
        wall_right = 1.0 if (c + 1 >= self.width or self.maze[r, c + 1] == 1) else 0.0
        wall_down = 1.0 if (r + 1 >= self.height or self.maze[r + 1, c] == 1) else 0.0
        wall_left = 1.0 if (c - 1 < 0 or self.maze[r, c - 1] == 1) else 0.0

        # Goal direction normalized
        dr = self.goal_pos[0] - r
        dc = self.goal_pos[1] - c
        norm = np.sqrt(dr**2 + dc**2) + 1e-6
        goal_dx = dc / norm
        goal_dy = dr / norm

        return np.array([wall_up, wall_right, wall_down, wall_left, goal_dx, goal_dy], dtype=np.float32)

    def step(self, action):
        r, c = self.agent_pos

        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = deltas[action]

        nr, nc = r + dr, c + dc

        reward = -0.02   # small negative to discourage pointless wandering
        terminated = False
        truncated = False

        # Check wall collision
        crashed = nr < 0 or nr >= self.height or nc < 0 or nc >= self.width or self.maze[nr, nc] == 1
        old_dist = abs(self.goal_pos[0] - r) + abs(self.goal_pos[1] - c)
        new_dist = abs(self.goal_pos[0] - nr) + abs(self.goal_pos[1] - nc)
        if crashed:
            reward -= 0.5  # crash penalty (kept small for EA stability)
        else:
            # Compute distance improvement
            if new_dist < old_dist:
                reward += 1.0 * (old_dist - new_dist)  # closer to goal
            elif new_dist > old_dist:
                reward -= 1.0 * (new_dist - old_dist)  # farther from goal

            self.agent_pos = (nr, nc)

            if self.agent_pos == self.goal_pos:
                reward += 100.0
                terminated = True

        # Align with precomputed shortest-step suggestion (A*-like guidance)
        best_act = self._best_action_map.get((self.agent_pos[0], self.agent_pos[1]))
        if best_act is not None and not terminated and not crashed:
            if action == best_act:
                reward += 2.0
            else:
                reward -= 0.5

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        info = {
            "crashed": crashed,
            "old_dist": old_dist,
            "new_dist": new_dist,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        import pygame

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size)
            )
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))
        canvas.fill((255, 255, 255))

        for r in range(self.height):
            for c in range(self.width):
                color = (255, 255, 255)
                if self.maze[r, c] == 1:
                    color = (0, 0, 0)
                elif self.maze[r, c] == 3:
                    color = (255, 215, 0)  # goal

                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size),
                )

        # Draw agent
        ar, ac = self.agent_pos
        pygame.draw.circle(
            canvas,
            (0, 128, 255),
            ((ac + 0.5) * self.cell_size, (ar + 0.5) * self.cell_size),
            self.cell_size // 3,
        )

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        import pygame
        if self.window:
            pygame.display.quit()
            pygame.quit()

    def _build_best_action_map(self):
        """
        BFS from goal to compute shortest path distances and best action toward goal for each cell.
        """
        dist = np.full((self.height, self.width), np.inf, dtype=np.float32)
        best_action_map = {}
        q = [self.goal_pos]
        dist[self.goal_pos] = 0.0
        head = 0
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while head < len(q):
            r, c = q[head]
            head += 1
            for a, (dr, dc) in enumerate(deltas):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and self.maze[nr, nc] != 1:
                    if dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        q.append((nr, nc))

        # For each cell, pick neighbor that reduces dist the most (greedy toward goal)
        for r in range(self.height):
            for c in range(self.width):
                if not np.isfinite(dist[r, c]):
                    continue
                best = None
                best_d = dist[r, c]
                for a, (dr, dc) in enumerate(deltas):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width and np.isfinite(dist[nr, nc]):
                        if dist[nr, nc] < best_d:
                            best_d = dist[nr, nc]
                            best = a
                if best is not None:
                    best_action_map[(r, c)] = best
        return best_action_map, dist


class FOVBasicEnv(gym.Env):
    """
    Minimal FoV maze environment:
    - 4 actions (Up/Right/Down/Left), single-cell moves
    - Observation: 8-tile FoV + orientation one-hot (no smell, no turn memory)
    - Rewards: small step penalty, wall crash penalty, goal reward
    - No smell shaping, no optimal-path penalties
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, maze_data, render_mode=None, step_penalty=-0.01, crash_penalty=-0.2, goal_reward=1.0):
        super().__init__()

        self.maze = maze_data
        self.height, self.width = self.maze.shape
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        obs_dim = 8 + 4  # FoV + orientation
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.start_pos = tuple(np.argwhere(self.maze == 2)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 3)[0])
        self.agent_pos = self.start_pos

        self.orientation = 1  # face right initially
        self.max_steps = max(10, self.height * self.width * 5)
        self.step_count = 0

        self.step_penalty = step_penalty
        self.crash_penalty = crash_penalty
        self.goal_reward = goal_reward

        self.window = None
        self.clock = None
        self.cell_size = 20

        self.grid = self.maze.copy()
        self.grid[self.start_pos] = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.orientation = 1
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        fov = self._get_fov_tiles()
        orientation = np.eye(4)[self.orientation].astype(np.float32)
        return np.concatenate([fov, orientation]).astype(np.float32)

    def step(self, action):
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        direction_idx = action % 4
        dr, dc = deltas[direction_idx]
        self.orientation = direction_idx

        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        reward = self.step_penalty
        terminated = False
        truncated = False

        crashed = not (0 <= nr < self.height and 0 <= nc < self.width) or self.grid[nr, nc] == 1
        if crashed:
            reward += self.crash_penalty
        else:
            self.agent_pos = (nr, nc)
            if self.agent_pos == self.goal_pos:
                reward += self.goal_reward
                terminated = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {"is_success": terminated}

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
                color = (255, 255, 255)
                if self.grid[r, c] == 1:
                    color = (0, 0, 0)
                elif (r, c) == self.goal_pos:
                    color = (255, 215, 0)
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size),
                )

        agent_color = (128, 128, 128)
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

    def _get_fov_tiles(self):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dir_vec = directions[self.orientation]
        left_vec = directions[(self.orientation - 1) % 4]
        right_vec = directions[(self.orientation + 1) % 4]

        def sample(pos):
            r, c = pos
            if not (0 <= r < self.height and 0 <= c < self.width):
                return 1.0
            if self.grid[r, c] == 1:
                return 1.0
            return 0.0

        x, y = self.agent_pos
        cl = sample((x + left_vec[0], y + left_vec[1]))
        cr = sample((x + right_vec[0], y + right_vec[1]))

        a1_pos = (x + dir_vec[0], y + dir_vec[1])
        a2_pos = (x + 2 * dir_vec[0], y + 2 * dir_vec[1])

        a1 = sample(a1_pos)
        a1l = sample((a1_pos[0] + left_vec[0], a1_pos[1] + left_vec[1]))
        a1r = sample((a1_pos[0] + right_vec[0], a1_pos[1] + right_vec[1]))

        a2 = sample(a2_pos)
        a2l = sample((a2_pos[0] + left_vec[0], a2_pos[1] + left_vec[1]))
        a2r = sample((a2_pos[0] + right_vec[0], a2_pos[1] + right_vec[1]))

        return np.array([cl, cr, a1, a1l, a1r, a2, a2l, a2r], dtype=np.float32)
