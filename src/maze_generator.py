import numpy as np
import random
import pandas as pd
import pickle
import os
from collections import deque
import heapq

# Constants
CELL_EMPTY = 0
CELL_WALL = 1
CELL_START = 2
CELL_GOAL = 3

# Action ordering aligns with MouseMazeEnvLegacy: 0=Up, 1=Right, 2=Down, 3=Left
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_NONE = -1

class MazeGenerator:
    def __init__(self, save_dir="data", include_flow_fields: bool = True):
        self.save_dir = save_dir
        # When True, also emit per-cell distance and action maps alongside paths
        self.include_flow_fields = include_flow_fields
        os.makedirs(save_dir, exist_ok=True)

    def generate_maze(self, width, height, complexity="perfect"):
        """
        Generates a maze.
        width, height: dimensions (must be odd numbers for wall logic).
        complexity: 'perfect' (DFS Backtracker) or 'imperfect' (loops allowed).
        """
        # Ensure odd dimensions
        width = width if width % 2 != 0 else width + 1
        height = height if height % 2 != 0 else height + 1
        
        maze = np.ones((height, width), dtype=np.int8) * CELL_WALL
        
        # Recursive Backtracker for Perfect Maze
        def get_neighbors(r, c):
            neighbors = []
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 < nr < height and 0 < nc < width and maze[nr, nc] == CELL_WALL:
                    neighbors.append((nr, nc))
            random.shuffle(neighbors)
            return neighbors

        start_node = (1, 1)
        maze[start_node] = CELL_EMPTY
        stack = [start_node]
        
        while stack:
            current = stack[-1]
            neighbors = get_neighbors(*current)
            
            if neighbors:
                next_node = neighbors[0]
                # Knock down wall between
                wall_r = (current[0] + next_node[0]) // 2
                wall_c = (current[1] + next_node[1]) // 2
                maze[wall_r, wall_c] = CELL_EMPTY
                maze[next_node] = CELL_EMPTY
                stack.append(next_node)
            else:
                stack.pop()
        
        # For Imperfect Maze: Randomly remove walls to create loops
        if complexity == "imperfect":
            num_loops = int((width * height) * 0.05) # remove 5% more walls
            for _ in range(num_loops):
                rx = random.randrange(1, height - 1)
                ry = random.randrange(1, width - 1)
                if maze[rx, ry] == CELL_WALL:
                    # Check if it connects two empty spaces
                    neighbors = 0
                    if maze[rx+1, ry] == CELL_EMPTY: neighbors += 1
                    if maze[rx-1, ry] == CELL_EMPTY: neighbors += 1
                    if maze[rx, ry+1] == CELL_EMPTY: neighbors += 1
                    if maze[rx, ry-1] == CELL_EMPTY: neighbors += 1
                    if neighbors >= 2:
                        maze[rx, ry] = CELL_EMPTY

        # Set Goal (furthest point usually, but random for diversity)
        maze[1, 1] = CELL_START
        maze[height-2, width-2] = CELL_GOAL
        
        return maze

    def solve_astar(self, maze):
        """
        Solves maze to find optimal steps (Ground Truth).
        """
        h, w = maze.shape
        start = tuple(np.argwhere(maze == CELL_START)[0])
        goal = tuple(np.argwhere(maze == CELL_GOAL)[0])
        
        queue = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        
        while queue:
            _, current = heapq.heappop(queue)
            
            if current == goal:
                break
            
            # Simple moves for ground truth (Up, Down, Left, Right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_node = (current[0] + dr, current[1] + dc)
                if 0 <= next_node[0] < h and 0 <= next_node[1] < w:
                    if maze[next_node] != CELL_WALL:
                        new_cost = cost_so_far[current] + 1
                        if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                            cost_so_far[next_node] = new_cost
                            priority = new_cost + abs(goal[0] - next_node[0]) + abs(goal[1] - next_node[1])
                            heapq.heappush(queue, (priority, next_node))
                            came_from[next_node] = current
                            
        # Reconstruct path
        if goal not in came_from:
            return None, 0 # Should not happen in generated mazes
            
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
        path.append(start)
        return path[::-1], len(path) - 1

    def compute_dist_action_maps(self, maze):
        """
        Compute a distance-to-goal field and greedy action map from any cell.
        Action order matches MouseMazeEnvLegacy (Up, Right, Down, Left).
        """
        h, w = maze.shape
        goal = tuple(np.argwhere(maze == CELL_GOAL)[0])

        dist_map = np.full((h, w), np.inf, dtype=np.float32)
        dist_map[goal] = 0.0
        queue = [(0.0, goal)]
        visited = set()

        while queue:
            curr_dist, (r, c) = heapq.heappop(queue)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # up, right, down, left
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] != CELL_WALL:
                    new_dist = curr_dist + 1
                    if new_dist < dist_map[nr, nc]:
                        dist_map[nr, nc] = new_dist
                        heapq.heappush(queue, (new_dist, (nr, nc)))

        action_map = np.full((h, w), ACTION_NONE, dtype=np.int8)
        for r in range(h):
            for c in range(w):
                if maze[r, c] == CELL_WALL or (r, c) == goal:
                    continue
                neighbors = [
                    (r - 1, c, ACTION_UP),
                    (r, c + 1, ACTION_RIGHT),
                    (r + 1, c, ACTION_DOWN),
                    (r, c - 1, ACTION_LEFT),
                ]
                best_action = ACTION_NONE
                best_dist = dist_map[r, c]
                for nr, nc, act in neighbors:
                    if 0 <= nr < h and 0 <= nc < w:
                        if dist_map[nr, nc] < best_dist:
                            best_dist = dist_map[nr, nc]
                            best_action = act
                action_map[r, c] = best_action

        finite = dist_map[np.isfinite(dist_map)]
        if finite.size > 0 and finite.max() > 0:
            norm_dist_map = 1.0 - (dist_map / finite.max())
            norm_dist_map[~np.isfinite(norm_dist_map)] = 0.0
        else:
            norm_dist_map = dist_map

        return norm_dist_map.astype(np.float32), action_map

    def _generate_level(self, level_name, width, height, total=4000):
        """
        Generate a full dataset for a single difficulty level.
        Saves into a subfolder named after dimensions, e.g., data/2121 for 21x21.
        """
        level_dir = os.path.join(self.save_dir, f"{height}{width}")
        os.makedirs(level_dir, exist_ok=True)

        per_type = total // 2
        records = []
        maze_id = 0

        for m_type, count in [("perfect", per_type), ("imperfect", total - per_type)]:
            print(f"Generating {count} mazes ({m_type}) for {level_name} ({height}x{width})...")
            for _ in range(count):
                maze = self.generate_maze(width, height, m_type)
                path, steps = self.solve_astar(maze)

                dist_map = None
                action_map = None
                if self.include_flow_fields:
                    dist_map, action_map = self.compute_dist_action_maps(maze)

                base = f"maze_{maze_id}"
                maze_file = f"{base}.npy"
                path_file = f"{base}_path.npy"
                dist_file = f"{base}_dist.npy" if self.include_flow_fields else None
                action_file = f"{base}_actions.npy" if self.include_flow_fields else None

                np.save(os.path.join(level_dir, maze_file), maze)
                np.save(os.path.join(level_dir, path_file), np.array(path, dtype=np.int16))
                if self.include_flow_fields and dist_map is not None and action_map is not None:
                    np.save(os.path.join(level_dir, dist_file), dist_map)
                    np.save(os.path.join(level_dir, action_file), action_map)

                records.append(
                    {
                        "id": maze_id,
                        "filename": maze_file,
                        "type": m_type,
                        "difficulty": level_name,
                        "height": height,
                        "width": width,
                        "optimal_steps": steps,
                        "path_file": path_file,
                        "path_len": steps,
                        "dist_file": dist_file,
                        "action_file": action_file,
                    }
                )
                maze_id += 1

        df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.9)
        df["split"] = "train"
        df.loc[split_idx:, "split"] = "test"
        meta_path = os.path.join(level_dir, "maze_metadata.csv")
        df.to_csv(meta_path, index=False)
        print(f"{level_name}: saved {len(df)} mazes to {level_dir} (metadata: {meta_path})")
        return df

    def generate_dataset(self):
        """
        Generate 3 datasets (Easy/Medium/Hard), each with 4000 mazes (half perfect, half imperfect).
        Stores each difficulty in its own subdirectory named by dimensions, e.g., data/2121 for Medium (21x21).
        """
        print("Generating datasets...")
        levels = [("Easy", 11, 11), ("Medium", 21, 21), ("Hard", 31, 31)]
        summaries = {}
        for name, h, w in levels:
            summaries[name] = self._generate_level(name, w, h, total=4000)
        return summaries

if __name__ == "__main__":
    gen = MazeGenerator()
    gen.generate_dataset()
