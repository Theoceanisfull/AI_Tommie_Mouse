import numpy as np
import random
import pandas as pd
import os
import heapq

# Constants
CELL_EMPTY = 0
CELL_WALL = 1
CELL_START = 2
CELL_GOAL = 3

# Action Constants for the Flow Field
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NONE = -1 # Walls or Goal

class MazeGenerator:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def generate_maze(self, width, height, complexity="perfect"):
        # Ensure odd dimensions
        width = width if width % 2 != 0 else width + 1
        height = height if height % 2 != 0 else height + 1
        
        maze = np.ones((height, width), dtype=np.int8) * CELL_WALL
        
        # Recursive Backtracker
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
                wall_r = (current[0] + next_node[0]) // 2
                wall_c = (current[1] + next_node[1]) // 2
                maze[wall_r, wall_c] = CELL_EMPTY
                maze[next_node] = CELL_EMPTY
                stack.append(next_node)
            else:
                stack.pop()
        
        # Imperfect Maze Logic
        if complexity == "imperfect":
            num_loops = int((width * height) * 0.05)
            for _ in range(num_loops):
                rx = random.randrange(1, height - 1)
                ry = random.randrange(1, width - 1)
                if maze[rx, ry] == CELL_WALL:
                    neighbors = 0
                    if maze[rx+1, ry] == CELL_EMPTY: neighbors += 1
                    if maze[rx-1, ry] == CELL_EMPTY: neighbors += 1
                    if maze[rx, ry+1] == CELL_EMPTY: neighbors += 1
                    if maze[rx, ry-1] == CELL_EMPTY: neighbors += 1
                    if neighbors >= 2:
                        maze[rx, ry] = CELL_EMPTY

        # Set Start and Goal
        maze[1, 1] = CELL_START
        maze[height-2, width-2] = CELL_GOAL
        
        return maze

    def solve_dijkstra_maps(self, maze):
        """
        Instead of a single path, this generates:
        1. Distance Map: Steps to goal from EVERY cell.
        2. Flow Field: The best action (0,1,2,3) to take from EVERY cell.
        This provides 'Recovery Data' for the neural network.
        """
        h, w = maze.shape
        goal = tuple(np.argwhere(maze == CELL_GOAL)[0])
        
        # 1. Initialize Distance Map (Infinity everywhere except goal)
        dist_map = np.full((h, w), 9999, dtype=np.float32)
        dist_map[goal] = 0
        
        # 2. Dijkstra / BFS from GOAL outwards
        queue = [(0, goal)]
        visited = set()
        
        while queue:
            curr_dist, (r, c) = heapq.heappop(queue)
            
            if (r,c) in visited: continue
            visited.add((r,c))
            
            # Check neighbors (Up, Down, Left, Right)
            # Note: We look for empty, start, or goal cells
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for nr, nc in neighbors:
                if 0 <= nr < h and 0 <= nc < w:
                    if maze[nr, nc] != CELL_WALL:
                        new_dist = curr_dist + 1
                        if new_dist < dist_map[nr, nc]:
                            dist_map[nr, nc] = new_dist
                            heapq.heappush(queue, (new_dist, (nr, nc)))
        
        # 3. Generate Flow Field (Action Map)
        # For every cell, look at neighbors and pick the one with lowest distance
        action_map = np.full((h, w), ACTION_NONE, dtype=np.int8)
        
        for r in range(h):
            for c in range(w):
                if maze[r, c] == CELL_WALL or (r,c) == goal:
                    continue
                
                # Check neighbors to find the one with lowest distance
                best_action = ACTION_NONE
                min_dist = dist_map[r, c] # Start with current distance
                
                # Check Up
                if r > 0 and dist_map[r-1, c] < min_dist:
                    min_dist = dist_map[r-1, c]
                    best_action = ACTION_UP
                # Check Down
                if r < h-1 and dist_map[r+1, c] < min_dist:
                    min_dist = dist_map[r+1, c]
                    best_action = ACTION_DOWN
                # Check Left
                if c > 0 and dist_map[r, c-1] < min_dist:
                    min_dist = dist_map[r, c-1]
                    best_action = ACTION_LEFT
                # Check Right
                if c < w-1 and dist_map[r, c+1] < min_dist:
                    min_dist = dist_map[r, c+1]
                    best_action = ACTION_RIGHT
                    
                action_map[r, c] = best_action
                
        # Normalize distance map for Neural Net training (0.0 to 1.0)
        # We invert it: 1.0 = Goal, 0.0 = Far away
        max_val = np.max(dist_map[dist_map != 9999])
        if max_val > 0:
            norm_dist_map = 1.0 - (dist_map / max_val)
            norm_dist_map[dist_map == 9999] = 0.0 # Walls
        else:
            norm_dist_map = dist_map

        return norm_dist_map, action_map

    def _generate_level(self, level_name, width, height, total=4000):
        level_dir = os.path.join(self.save_dir, f"{height}{width}")
        os.makedirs(level_dir, exist_ok=True)

        per_type = total // 2
        records = []
        maze_id = 0

        for m_type, count in [("perfect", per_type), ("imperfect", total - per_type)]:
            print(f"Generating {count} mazes ({m_type}) for {level_name} ({height}x{width})...")
            for _ in range(count):
                maze = self.generate_maze(width, height, m_type)
                
                # NEW: Get maps instead of a single path
                dist_map, action_map = self.solve_dijkstra_maps(maze)

                base = f"maze_{maze_id}"
                
                # Save 3 files: The Maze, The Scent (Distance), The Teacher (Actions)
                np.save(os.path.join(level_dir, f"{base}_maze.npy"), maze)
                np.save(os.path.join(level_dir, f"{base}_dist.npy"), dist_map)
                np.save(os.path.join(level_dir, f"{base}_actions.npy"), action_map)

                records.append(
                    {
                        "id": maze_id,
                        "type": m_type,
                        "difficulty": level_name,
                        "height": height,
                        "width": width,
                        "maze_file": f"{base}_maze.npy",
                        "dist_file": f"{base}_dist.npy",
                        "action_file": f"{base}_actions.npy",
                    }
                )
                maze_id += 1

        df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
        # Create CSV metadata
        split_idx = int(len(df) * 0.9)
        df["split"] = "train"
        df.loc[split_idx:, "split"] = "test"
        
        meta_path = os.path.join(level_dir, "maze_metadata.csv")
        df.to_csv(meta_path, index=False)
        print(f"{level_name}: saved {len(df)} mazes to {level_dir}")
        return df

    def generate_dataset(self):
        print("Generating datasets with Distance Maps and Flow Fields...")
        levels = [("Easy", 11, 11), ("Medium", 21, 21), ("Hard", 31, 31)]
        for name, h, w in levels:
            self._generate_level(name, w, h, total=4000)

if __name__ == "__main__":
    gen = MazeGenerator()
    gen.generate_dataset()