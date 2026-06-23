import numpy as np
import heapq

class AStarPlanner:
    def __init__(self, x_min=-2.5, x_max=2.5, y_min=-2.5, y_max=2.5, resolution=0.10, safety_margin=0.22):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        self.safety_margin = safety_margin
        
        self.nx = int((x_max - x_min) / resolution)
        self.ny = int((y_max - y_min) / resolution)
        
    def _world_to_grid(self, x, y):
        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)
        gx = max(0, min(gx, self.nx - 1))
        gy = max(0, min(gy, self.ny - 1))
        return gx, gy
        
    def _grid_to_world(self, gx, gy):
        wx = self.x_min + (gx + 0.5) * self.resolution
        wy = self.y_min + (gy + 0.5) * self.resolution
        return wx, wy

    def plan_on_grid(self, start, goal, grid):
        """
        Plan a path on a pre-populated grid.
        grid is a 2D numpy array of shape (nx, ny) where 1 is occupied, 0 is free.
        """
        start_grid = self._world_to_grid(start[0], start[1])
        goal_grid = self._world_to_grid(goal[0], goal[1])
        
        # Ensure start, a 3x3 footprint around start, and goal are free in the grid copy
        grid_copy = np.copy(grid)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx_c = start_grid[0] + dx
                ny_c = start_grid[1] + dy
                if 0 <= nx_c < self.nx and 0 <= ny_c < self.ny:
                    grid_copy[nx_c, ny_c] = 0
        grid_copy[goal_grid[0], goal_grid[1]] = 0
        
        open_set = []
        heapq.heappush(open_set, (0.0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0.0}
        f_score = {start_grid: np.linalg.norm(np.array(start_grid) - np.array(goal_grid))}
        
        motions = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, 1.414),
            (1, -1, 1.414),
            (-1, 1, 1.414),
            (-1, -1, 1.414)
        ]
        
        found = False
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                found = True
                break
                
            for dx, dy, cost in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.nx and 0 <= neighbor[1] < self.ny):
                    continue
                if grid_copy[neighbor[0], neighbor[1]] == 1:
                    continue
                    
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = np.linalg.norm(np.array(neighbor) - np.array(goal_grid))
                    f_score[neighbor] = tentative_g + h
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        if not found:
            return [start, goal], False
            
        path_grid = [goal_grid]
        curr = goal_grid
        while curr in came_from:
            curr = came_from[curr]
            path_grid.append(curr)
        path_grid.reverse()
        
        path_world = []
        for gx, gy in path_grid:
            path_world.append(self._grid_to_world(gx, gy))
            
        path_world[0] = (start[0], start[1])
        path_world[-1] = (goal[0], goal[1])
        return path_world, True
        
    def plan(self, start, goal, walls):
        # 1. Build occupancy grid from walls
        grid = np.zeros((self.nx, self.ny), dtype=np.int8)
        for gx in range(self.nx):
            for gy in range(self.ny):
                wx, wy = self._grid_to_world(gx, gy)
                for wx_wall, wy_wall, hx, hy, yaw in walls:
                    dx = wx - wx_wall
                    dy = wy - wy_wall
                    cos_y = np.cos(yaw)
                    sin_y = np.sin(yaw)
                    local_x = dx * cos_y + dy * sin_y
                    local_y = -dx * sin_y + dy * cos_y
                    
                    if abs(local_x) <= hx + self.safety_margin and abs(local_y) <= hy + self.safety_margin:
                        grid[gx, gy] = 1
                        break
                        
        return self.plan_on_grid(start, goal, grid)[0]
