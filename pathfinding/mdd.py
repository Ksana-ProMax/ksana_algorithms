"""
Multi-Valued Decision Diagram (MDD)

The increasing cost tree search for optimal multi-agent pathfinding

"""
from dataclasses import dataclass
from typing import Literal, Optional, Self
import numpy

Point = tuple[int, int]


@dataclass
class Node:
    position: Point
    """agent所处的位置"""
    timestamp: int
    """时间点t, 同时也是层数"""
    parents: list[Self]
    """父节点"""
    children: list[Self]
    """节点的子节点"""

    def __repr__(self):
        return f"Node(pos={self.position}, t={self.timestamp})"


class MDD:
    def __init__(self,
                 grid: numpy.ndarray,
                 cost: int,
                 move: Literal["4way", "8way"] = "4way") -> None:
        """
        Args:
            - grid : 场地, 0代表可以移动的区域, -1代表不可移动的区域, 1代表起点, 2代表终点
            - move : 移动方式, 4way代表上下左右四个方向, 8way代表在4way上加入四个角方向上的移动
            - cost : 路径成本
        """
        self.nodes = []
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = tuple([int(elem) for elem in numpy.where(grid == 1)])
        self.dest = tuple([int(elem) for elem in numpy.where(grid == 2)])

        if self.start is None or self.dest is None:
            return None

        self.move = move
        if move == "4way":
            self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        elif move == "8way":
            self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 上、下、左、右
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 四个对角线方向
        else:
            return None

        self.cost = cost
        self.build_mdd()

    def build_mdd(self):
        """利用BFS构造MDD"""
        assert len(self.start) == 2 and len(self.dest) == 2     # 避免语法问题
        start_node = Node(self.start, 0, [], [])

        all_layer_nodes: list[list[Node]] = [[start_node]]  # 分层的所有节点，每一层是一个 list
        layer = [start_node]
        for t in range(self.cost):
            next_layer: list[Node] = []
            for node_t in layer:
                pos = node_t.position
                for dx, dy in self.dirs:
                    nx, ny = pos[0] + dx, pos[1] + dy

                    if (0 <= nx < self.rows) and (0 <= ny < self.cols) and (self.grid[nx, ny] != -1):
                        next_pos = (nx, ny)

                        node_t_plus_1 = next((node for node in next_layer if node.position == next_pos), None)
                        if node_t_plus_1 is None:    # 创建节点
                            node_t_plus_1 = Node(next_pos, t + 1, [node_t], [])
                            next_layer.append(node_t_plus_1)  # 加入到下一层中
                        else:    # 获取现有节点
                            if node_t not in node_t_plus_1.parents:
                                node_t_plus_1.parents.append(node_t)

                        if node_t_plus_1 not in node_t.children:
                            node_t.children.append(node_t_plus_1)
            layer = next_layer
            all_layer_nodes.append(layer)

        goal = next((node for node in layer if node.position == self.dest), None)

        if goal is None:
            self.nodes = []
        else:
            # 最后一层只保留 goal 节点
            keep_nodes: list[list[Node]] = []
            nodes: list[Node] = [goal]
            for t in range(self.cost+1):
                parents = []
                keep_nodes.append(nodes)
                for node in nodes:
                    parent_nodes = node.parents
                    parents += [node for node in parent_nodes if node not in parents]
                nodes = parents
            self.nodes = keep_nodes[::-1]


if __name__ == "__main__":
    grid = numpy.array([
        [1, 0, 0, 0, -1],
        [0, -1, -1, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, 0, -1, -1],
        [0, -1, 0, 0, 2]
    ], dtype=int)
    mdd = MDD(grid, 10, "4way")
    print(grid, "\n", mdd.nodes)
