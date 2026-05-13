import heapq
from typing import List, Tuple

import numpy as np


class AStarPathPlanner:
    """2D grid A* planner with obstacle inflation for wall avoidance."""

    def __init__(self, world_limit: float = 2.5, resolution: float = 0.10, inflation: float = 0.18):
        self.world_limit = world_limit
        self.resolution = resolution
        self.inflation = inflation
        self.grid_size = int((2 * world_limit) / resolution) + 1

    def _to_grid(self, xy: np.ndarray) -> Tuple[int, int]:
        x = int(round((xy[0] + self.world_limit) / self.resolution))
        y = int(round((xy[1] + self.world_limit) / self.resolution))
        return max(0, min(self.grid_size - 1, x)), max(0, min(self.grid_size - 1, y))

    def _to_world(self, ij: Tuple[int, int]) -> np.ndarray:
        x = ij[0] * self.resolution - self.world_limit
        y = ij[1] * self.resolution - self.world_limit
        return np.array([x, y], dtype=np.float32)

    def _heur(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _build_occupancy(self, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        occ = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        inflate = self.inflation
        for c_xy, radius in obstacles:
            r = radius + inflate
            min_pt = self._to_grid(c_xy - r)
            max_pt = self._to_grid(c_xy + r)
            for i in range(min_pt[0], max_pt[0] + 1):
                for j in range(min_pt[1], max_pt[1] + 1):
                    w = self._to_world((i, j))
                    if np.linalg.norm(w - c_xy) <= r:
                        occ[i, j] = 1
        return occ

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> List[np.ndarray]:
        start = self._to_grid(start_xy)
        goal = self._to_grid(goal_xy)
        occ = self._build_occupancy(obstacles)

        occ[start[0], start[1]] = 0
        occ[goal[0], goal[1]] = 0

        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]

        pq = []
        heapq.heappush(pq, (0.0, start))
        came = {}
        g = {start: 0.0}

        while pq:
            _, cur = heapq.heappop(pq)
            if cur == goal:
                break
            for di, dj, cost in moves:
                ni, nj = cur[0] + di, cur[1] + dj
                if ni < 0 or nj < 0 or ni >= self.grid_size or nj >= self.grid_size:
                    continue
                if occ[ni, nj] == 1:
                    continue
                nxt = (ni, nj)
                ng = g[cur] + cost
                if nxt not in g or ng < g[nxt]:
                    g[nxt] = ng
                    f = ng + self._heur(nxt, goal)
                    heapq.heappush(pq, (f, nxt))
                    came[nxt] = cur

        if goal not in came and goal != start:
            return [goal_xy.astype(np.float32)]

        path_grid = [goal]
        while path_grid[-1] != start:
            path_grid.append(came[path_grid[-1]])
        path_grid.reverse()

        # Downsample waypoints to reduce jitter.
        path = [self._to_world(p) for p in path_grid]
        if len(path) <= 2:
            return [goal_xy.astype(np.float32)]
        sampled = [path[0]]
        for k in range(2, len(path), 3):
            sampled.append(path[k])
        sampled.append(path[-1])
        return [p.astype(np.float32) for p in sampled]
