"""
支持多个 agent 的 CBS 代码, 
由 Deepseek 进行的代码改写
用于 MACBS 的 joint 求解
"""

import copy
from dataclasses import dataclass
import heapq
from typing import Optional, Self
import numpy

Point = tuple[int, int]


@dataclass
class Constraint:
    agent: int
    """发生冲突的agent, 索引形式"""
    coord: Point
    """不能出现的地点"""
    time: int
    """不能出现的时间点t"""


@dataclass
class Node:
    constraints: list
    cost: int
    solution: dict[int, list[Point]]
    goal: bool
    parent: Optional[Self] = None
    child: Optional[Self] = None

    def __lt__(self, other):
        """heap使用时的比较函数"""
        return self.cost < other.cost


class CBSMulti:
    def __init__(self, grid: numpy.ndarray,
                 starts: list[Point],
                 dests: list[Point],
                 move_type: str = "4way"):
        """
        Args:
            - grid : 场地, 0代表可以移动的区域, -1代表不可移动的区域
            - starts : 起点位置
            - dests : 终点位置
        """
        self.grid: numpy.ndarray = grid
        self.rows, self.cols = grid.shape
        assert self.rows > 0 and self.cols > 0

        self.starts: list[Point] = starts
        self.dests: list[Point] = dests
        assert len(self.dests) == len(self.starts)

        # 检查是否越界和是否处于不可移动区域
        for (x, y) in starts + dests:
            if not (0 <= x < self.rows and 0 <= y < self.cols) or (self.grid[x, y] == -1):
                raise Exception("节点不合法")

        self.num_agents = len(starts)

        # 移动方向
        if move_type == "4way":
            # 上、下、左、右, 以及允许原地停留
            self.dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            # 上、下、左、右, 以及对角线，以及允许原地停留
            self.dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def high_level_search(self) -> Optional[dict[int, list[Point]]]:
        """
        参见论文里面的 Algorithm 1
        以map形式返回每个 agent 的最佳路径
        """
        open = []

        solution = {}
        cost = 0
        for agent in range(self.num_agents):
            path = self.low_level_search(agent, [])  # root 节点没有任何的约束
            if path is None:
                return None  # 无解
            solution[agent] = path
            cost += (len(path) - 1)

        root = Node(constraints=[], cost=cost, solution=solution, goal=False)
        heapq.heappush(open, root)

        index = 0
        while open:
            index += 1

            current_node: Node = heapq.heappop(open)  # 获取 cost 最小的节点
            conflict = self.validate(current_node)  # 校验解决方案中是否有冲突

            if not conflict:
                current_node.goal = True
                return current_node.solution
            else:
                agents_in_conflict, coord, time = conflict

                # 取前两个智能体进行分支
                # only focus on the first two agents that are found to conflict
                agent_i, agent_j = agents_in_conflict[0], agents_in_conflict[1]

                for agent in [agent_i, agent_j]:
                    constraints = current_node.constraints + [Constraint(agent, coord, time)]

                    new_solution = copy.deepcopy(current_node.solution)
                    path = self.low_level_search(agent, constraints)
                    if path is not None:
                        new_solution[agent] = path
                        cost = sum(len(path) - 1 for path in new_solution.values())
                        new_node = Node(constraints=constraints, cost=cost,
                                        solution=new_solution, goal=False, parent=current_node)
                        heapq.heappush(open, new_node)

            print(f"当前搜索次数: {index}, heap大小: {len(open)}", end="\r")
        return None  # 无解

    def low_level_search(self,
                         agent: int,
                         constraints: list[Constraint]) -> Optional[list[Point]]:
        start = self.starts[agent]
        dest = self.dests[agent]

        open_list = []
        heapq.heappush(open_list, (0, 0, start, 0, [start]))  # (f, g, position, time, path)

        visited = {}

        index = 0
        while open_list:
            index += 1
            f, g, current, time, path = heapq.heappop(open_list)

            if current == dest:
                return path

            next_time = time + 1

            moves = [current]
            for dx, dy in self.dirs:
                nx, ny = current[0] + dx, current[1] + dy
                # 没有越界, 且地形为可移动
                if (0 <= nx < self.rows and 0 <= ny < self.cols and self.grid[nx, ny] == 0):
                    moves.append((nx, ny))

            # 对于所有的移动方式, 检测其是否符合约束条件
            for move in moves:
                constraint_violated = False
                for constraint in constraints:
                    if (constraint.agent == agent and constraint.coord == move and constraint.time == next_time):
                        constraint_violated = True
                        break

                # 如果违反了约束, 那么不能执行指定的移动
                if constraint_violated:
                    continue

                new_g = g + 1
                new_path = path + [move]

                h = abs(move[0] - dest[0]) + abs(move[1] - dest[1])  # heuristic, 曼哈顿距离
                new_f = new_g + h

                state_key = (move, next_time)
                if state_key in visited and visited[state_key] <= new_g:
                    continue

                visited[state_key] = new_g
                heapq.heappush(open_list, (new_f, new_g, move, next_time, new_path))
        return None

    def validate(self, node: Node) -> Optional[tuple]:
        solution = node.solution
        max_time = max(len(path) for path in solution.values())

        for t in range(max_time):
            occupied = {}   # 在时间点t时, 被占领的格子

            for agent, path in solution.items():
                if t < len(path):
                    pos = path[t]   # 获取 t 时间点时, agent 所处的位置
                    if pos in occupied:
                        # 如果所处的位置被其他 agent 所占领, 返回所有冲突的智能体
                        agents_in_conflict = [occupied[pos], agent]
                        return (agents_in_conflict, pos, t)
                    occupied[pos] = agent

        return None

    def visualize(self, solution: Optional[dict[int, list[Point]]]):
        """
        Deepseek 写的格式化输出
        """
        if not solution:
            print("无解")
            return

        max_path_length = max(len(path) for path in solution.values())

        time_grids = []
        for t in range(max_path_length):
            time_grid = numpy.full((self.rows, self.cols), "   ", dtype=object)

            for i in range(self.rows):
                for j in range(self.cols):
                    if self.grid[i, j] == -1:
                        time_grid[i, j] = "###"

            for agent, path in solution.items():
                # 如果路径已经结束，显示在目标位置
                if t >= len(path):
                    pos = path[-1]  # 保持在终点
                else:
                    pos = path[t]
                time_str = f"[{agent}]"
                time_grid[pos] = time_str

            time_grids.append(time_grid)

        for t, grid in enumerate(time_grids):
            print("-"*20)
            print(f"第{t}步:")
            for i in range(self.rows):
                row = ""
                for j in range(self.cols):
                    row += grid[i, j] + " "
                print(row)
            print()

        for agent, path in solution.items():
            print(f"角色 {agent}: {path}")


if __name__ == "__main__":
    grid = numpy.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)
    starts = [(0, 0), (0, 4), (2, 4), (2, 0)]
    dests = [(0, 4), (0, 0), (2, 0), (2, 4)]

    cbs = CBSMulti(grid, starts, dests, "4way")
    solution = cbs.high_level_search()
    cbs.visualize(solution)
