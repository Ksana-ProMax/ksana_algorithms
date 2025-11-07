"""
利用BFS来搜索最短路径
"""

from collections import deque
from typing import Literal, Optional
import numpy

Point = tuple[int, int]


def bfs_find_path(grid: numpy.ndarray, move: Literal["4way", "8way"]) -> Optional[list[Point]]:
    """
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

    queue: deque[Point] = deque([start])
    traj: dict[Point, Point] = {}

    while True:
        if len(queue) <= 0:
            break

        current = queue.popleft()
        if current == dest:
            path = []

            # 根据路径跳转表一直往上追溯
            while current != start:
                path.append(current)
                current = traj[current]
            path.append(start)
            return path

        # 遍历所有可能的方向, 将下一个未探索的节点追加到 queue 里面
        for dx, dy in dirs:
            next_x, next_y = current[0] + dx, current[1] + dy

            # 条件: 未越界, 未探索, 节点可行(值不为-1)
            if 0 <= next_x < rows and 0 <= next_y < cols:
                if grid[next_x, next_y] != -1 and (next_x, next_y) not in traj:
                    traj[(next_x, next_y)] = current
                    queue.append((next_x, next_y))
    return None


if __name__ == "__main__":
    grid = numpy.array([
        [1, 0, 0, 0, -1],
        [0, -1, -1, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, 0, -1, -1],
        [0, -1, 0, 0, 2]
    ], dtype=int)
    path = bfs_find_path(grid, "4way")
    print(grid, "\n", path)
