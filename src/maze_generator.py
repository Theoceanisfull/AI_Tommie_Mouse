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

class MazeGenerator:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
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

    def generate_dataset(self):
        print("Generating Dataset...")
        dataset = []
        configs = [
            ("Easy", 11, 11),
            ("Medium", 21, 21),
            ("Hard", 31, 31)
        ]
        
        # 2000 Perfect, 2000 Imperfect
        # Split evenly across Easy/Med/Hard
        
        counts = {"perfect": 2000, "imperfect": 2000}
        
        all_mazes = {}

        maze_id = 0
        for m_type, total in counts.items():
            per_level = total // 3
            for level, h, w in configs:
                print(f"Generating {m_type} - {level}...")
                for _ in range(per_level):
                    maze = self.generate_maze(w, h, m_type)
                    path, steps = self.solve_astar(maze)
                    path_filename = f"maze_{maze_id}_path.npy"
                    np.save(os.path.join(self.save_dir, path_filename), np.array(path, dtype=np.int16))
                    
                    filename = f"maze_{maze_id}.npy"
                    np.save(os.path.join(self.save_dir, filename), maze)
                    
                    all_mazes[maze_id] = maze
                    
                    dataset.append({
                        "id": maze_id,
                        "filename": filename,
                        "type": m_type,
                        "difficulty": level,
                        "height": h,
                        "width": w,
                        "optimal_steps": steps,
                        "path_file": path_filename
                    })
                    maze_id += 1
        
        # Shuffle once and create an 80/20 train/test split
        df = pd.DataFrame(dataset).sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        df["split"] = "train"
        df.loc[split_idx:, "split"] = "test"

        df.to_csv(os.path.join(self.save_dir, "maze_metadata.csv"), index=False)
        print(f"Dataset generated with {len(df)} mazes.")
        return df

if __name__ == "__main__":
    gen = MazeGenerator()
    gen.generate_dataset()
