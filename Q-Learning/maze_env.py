import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from PIL import Image, ImageDraw, ImageFont
import io

class MazeEnv(gym.Env):
    """
    Maze Environment (Gymnasium API)
    - Supports very large mazes (auto-scaled)
    - Emoji rendering (🐭, 🧀) with safe fallback
    - Purple wall rendering
    - Minimal reward inside environment
    - Proper info dictionary exposing hit_wall, step_count, etc.
    """

    metadata = {"render_modes": ["human", "ascii"], "render_fps": 10}

    # ----------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------
    def __init__(self, maze: np.ndarray, max_steps=None, render_mode=None):
        super().__init__()

        assert maze.ndim == 2, "Maze must be a 2D numpy array"
        self.maze = maze.astype(int)
        self.n_rows, self.n_cols = self.maze.shape

        # Locate start/goal markers if present (2=start, 3=goal), otherwise default corners
        start_cells = np.argwhere(self.maze == 2)
        goal_cells = np.argwhere(self.maze == 3)
        self.start_pos = tuple(start_cells[0]) if len(start_cells) else (0, 0)
        self.goal_pos = tuple(goal_cells[0]) if len(goal_cells) else (self.n_rows - 1, self.n_cols - 1)

        # Treat markers as free space
        self.maze[self.start_pos] = 0
        self.maze[self.goal_pos] = 0

        # Gymnasium spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)

        # Episode control (allow more steps for larger mazes)
        self.max_steps = max_steps or (self.n_rows * self.n_cols * 6)
        self.current_steps = 0
        self.agent_pos = None

        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.mouse_surface = None
        self.cheese_surface = None

    # ----------------------------------------------------------
    # Helper conversion
    # ----------------------------------------------------------
    def _pos_to_state(self, pos):
        r, c = pos
        return r * self.n_cols + c

    def _is_free(self, r, c):
        return (
            0 <= r < self.n_rows
            and 0 <= c < self.n_cols
            and self.maze[r, c] == 0
        )

    # ----------------------------------------------------------
    # Safe emoji rendering using PIL + pygame
    # ----------------------------------------------------------
    def _make_emoji_surface(self, emoji: str, size: int):
        """Generate an emoji surface with fallback."""
        size = max(size, 8)
        font = None

        # Try common emoji fonts
        for f in ["Apple Color Emoji.ttc", "Segoe UI Emoji.ttf", "NotoColorEmoji.ttf"]:
            try:
                font = ImageFont.truetype(f, size)
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()

        img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        try:
            bbox = draw.textbbox((0, 0), emoji, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textlength(emoji, font=font), size

        draw.text(((size - w) / 2, (size - h) / 2),
                  emoji, font=font, fill=(0, 0, 0, 255))

        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            buffer.seek(0)
            surface = pygame.image.load(buffer, "emoji.png").convert_alpha()

        return pygame.transform.scale(surface, (size, size))

    # ----------------------------------------------------------
    # Reset environment
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.current_steps = 0

        obs = self._pos_to_state(self.agent_pos)
        info = {
            "hit_wall": False,
            "step_count": 0,
            "position": self.agent_pos,
            "goal_pos": self.goal_pos,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    # ----------------------------------------------------------
    # Step environment
    # ----------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action)

        self.current_steps += 1
        r, c = self.agent_pos

        # Up, Right, Down, Left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        nr, nc = r + moves[action][0], c + moves[action][1]

        hit_wall = False
        if self._is_free(nr, nc):
            self.agent_pos = (nr, nc)
        else:
            hit_wall = True

        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self.current_steps >= self.max_steps and not terminated)

        # Reward shaping: step cost, crash penalty, progress bonus, goal bonus
        reward = -0.01  # step cost (lighter to avoid large negatives on long paths)
        if hit_wall:
            reward -= 1.0
        else:
            # Manhattan distance to goal
            old_dist = abs(self.goal_pos[0] - r) + abs(self.goal_pos[1] - c)
            new_dist = abs(self.goal_pos[0] - self.agent_pos[0]) + abs(self.goal_pos[1] - self.agent_pos[1])
            if new_dist < old_dist:
                reward += 0.2
            elif new_dist > old_dist:
                reward -= 0.1
        if terminated:
            reward += 20.0
        obs = self._pos_to_state(self.agent_pos)

        info = {
            "hit_wall": hit_wall,
            "step_count": self.current_steps,
            "position": self.agent_pos,
            "goal_pos": self.goal_pos,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------
    # Auto-scaled rendering
    # ----------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "ascii":
            print(self._ascii_render())
            return

        # ---------- Auto-scale ----------
        max_dim = max(self.n_rows, self.n_cols)
        max_pixels = 900
        cell = max(4, max_pixels // max_dim)

        width, height = self.n_cols * cell, self.n_rows * cell

        # ---------- Initialize pygame ----------
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("🐭 Maze Environment 🧀")
            self.clock = pygame.time.Clock()

            self.mouse_surface = self._make_emoji_surface("🐭", cell - 2)
            self.cheese_surface = self._make_emoji_surface("🧀", cell - 2)

        PURPLE = (160, 32, 240)
        WHITE = (255, 255, 255)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.window.fill(WHITE)

        # Draw walls
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.maze[r, c] == 1:
                    pygame.draw.rect(
                        self.window, PURPLE,
                        (c * cell, r * cell, cell, cell)
                    )

        # Goal
        gr, gc = self.goal_pos
        self.window.blit(self.cheese_surface, (gc * cell, gr * cell))

        # Agent
        ar, ac = self.agent_pos
        self.window.blit(self.mouse_surface, (ac * cell, ar * cell))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    # ----------------------------------------------------------
    # ASCII fallback (optional)
    # ----------------------------------------------------------
    def _ascii_render(self):
        chars = []
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                if (r, c) == self.agent_pos:
                    row += "A"
                elif (r, c) == self.goal_pos:
                    row += "G"
                else:
                    row += "#" if self.maze[r, c] == 1 else "."
            chars.append(row)
        return "\n".join(chars)

    # ----------------------------------------------------------
    # Close pygame
    # ----------------------------------------------------------
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
