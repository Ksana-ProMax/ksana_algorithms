"""
利用Dijkstra算法来搜索最短路径
"""

import heapq
import math
from typing import Literal, Optional
import numpy


Point = tuple[int, int]


def d_find_path(grid: numpy.ndarray, move: Literal["4way", "8way"]) -> Optional[list[Point]]:
    """
    Dijkstra算法, 每次获取最小值节点，然后更新与之相连的其他节点

    Args:
        - grid : 场地, 0代表可以移动的区域, -1代表不可移动的区域, 1代表起点, 2代表终点
        - move : 移动方式, 4way代表上下左右四个方向, 8way代表在4way上加入四个角方向上的移动

    Returns:
        路径的所有节点
        None 代表没有找到合适的路径
    """
    rows, cols = grid.shape
    start = tuple([int(elem) for elem in numpy.where(grid == 1)])
    dest = tuple([int(elem) for elem in numpy.where(grid == 2)])
    assert len(start) == 2 and len(dest) == 2

    if start is None or dest is None:
        return None

    if move == "4way":
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
    elif move == "8way":
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 上、下、左、右
                (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 四个对角线方向
    else:
        return None

    # 节点的值，初始化所有节点值为 -无穷，除了起点的值为0
    dist = numpy.full((rows, cols), numpy.inf)
    dist[start] = 0

    prev = {start: None}    # 跳转表
    heap: list[tuple[float, Point]] = [(0.0, start)]  # binary-heap
    visited = numpy.zeros((rows, cols), dtype=bool)

    while heap:
        # 每次获取的都是最小值的节点
        cost, (r, c) = heapq.heappop(heap)

        # 如果当前节点已经访问过，
        # 那么跳过这个节点，获取下一个节点
        if visited[r, c]:
            continue
        visited[r, c] = True

        # 到达终点，根据prev跳转表来重建路径
        if (r, c) == dest:
            path: list[Point] = []
            cur: Optional[Point] = dest
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            return path

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == -1:
                continue

            # chatgpt 给出的距离计算公式
            step_cost = math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1.0
            new_cost = cost + step_cost

            # 发现更短路径，更新对应的节点值
            if new_cost < dist[nr, nc]:
                dist[nr, nc] = new_cost
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost, (nr, nc)))

    # 没有找到路径
    return None


if __name__ == "__main__":
    grid = numpy.array([
        [1, 0, 0, 0, -1],
        [0, -1, -1, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, 0, -1, -1],
        [0, -1, 0, 0, 2]
    ], dtype=int)
    path = d_find_path(grid, "4way")
    print(f"{grid}, \n最短步数: {len(path)-1 if path is not None else "无合适路径"}, {path}")
