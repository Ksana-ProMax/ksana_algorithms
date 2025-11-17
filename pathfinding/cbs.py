"""
Conflict-Based Search For Optimal Multi-Agent Path Finding
Guni Sharon, Roni Stern, Ariel Felner, Nathan Sturtevant

注释里面，列表形式的注释，为论文的原文
"""

import copy
from dataclasses import dataclass
import heapq
from typing import Optional, Self

import numpy

Point = tuple[int, int]


@dataclass
class Constraint:
    """
    路径约束

    - The key idea of CBS is to grow a set of constraints for each of the agents and 
    find paths that are consistent with these constraints.
    """
    agent: int
    """发生冲突的agent, 索引形式"""
    coord: Point
    """不能出现的地点"""
    time: int
    """不能出现的时间点t"""


@dataclass
class Node:
    """
    Constraint Tree里面的一个节点
    """
    constraints: list
    """
    路径约束列表
    - The root of the CT contains an empty set of constraints. The child of a node in the CT 
    inherits the constraints of the parent and adds one new constraint for one agent.
    """
    cost: int
    """
    所有agent的总成本
    - The total cost (N:cost) of the current solution (summation over all the single-agent path costs).
    We denote this cost the f-value of the node.
    """
    solution: dict[int, list[Point]]
    """
    每个agent的路径
    - A set of k paths, one path for each agent. The path for agent ai must be consistent with the constraints 
    of ai. Such paths are found by the lowlevel
    """
    conflicts: list[tuple]
    """
    当前solution下的所有conflict, 为后续的优化算法做准备
    """
    goal: bool
    """
    当前的节点是否为goal. 
    如果所有的agent都没有冲突地抵达目的地, 那么这个节点就是goal
    - Node N in the CT is a goal node when N.solution is valid, i.e., the set of paths for 
    all agents have no conflicts.
    """
    parent: Optional[Self] = None
    child: Optional[Self] = None

    def __lt__(self, other):
        """heap使用时的比较函数"""
        return self.cost < other.cost


class CBS:
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
        assert len(self.starts) == 2 and len(self.dests) == 2
        # 目前只支持2个agent
        # There are two ways to handle such k-agent conflicts. We can generate k children,
        # each of which adds a constraint to k 􀀀 1 agents (i.e., each child allows only one agent to occupy the
        # conflicting vertex v at time t). Or, an equivalent formalization is to only focus on the first two agents
        # that are found to conflict, and only branch according to their conflict. This leaves further conflicts
        # for deeper levels of the tree.

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

        conflicts = self.validate(solution)
        root = Node(constraints=[], cost=cost, solution=solution, conflicts=conflicts, goal=False)
        heapq.heappush(open, root)

        index = 0
        while open:
            index += 1
            current_node: Node = heapq.heappop(open)  # 获取 cost 最小的节点

            if not current_node.conflicts:
                current_node.goal = True
                return current_node.solution
            else:
                conflict = current_node.conflicts[0]    # 选择第一个作为冲突
                agent_i, agent_j, coord, time = conflict

                # 将 conflict 分为两部分, 左侧代表了对于 agent_i 的约束条件, 右侧代表了对于 agent_j 的约束条件
                for agent, coord, t in [(agent_i, coord, time), (agent_j, coord, time)]:
                    constraints = current_node.constraints + [Constraint(agent, coord, t)]
                    path = self.low_level_search(agent, constraints)
                    if path is not None:
                        solution = copy.deepcopy(current_node.solution)
                        solution[agent] = path
                    cost = sum(len(path) - 1 for path in solution.values())
                    conflicts = self.validate(solution)
                    new_node = Node(constraints=constraints, cost=cost,
                                    solution=solution, conflicts=conflicts, goal=False, parent=current_node)
                    heapq.heappush(open, new_node)

            print(f"当前搜索次数: {index}, heap大小: {len(open)}", end="\r")
        return None  # 无解

    def low_level_search(self,
                         agent: int,
                         constraints: list[Constraint]) -> Optional[list[Point]]:
        """
        基于 A* 寻路算法的low-level search, 寻路时忽视其他的智能体。这里的寻路算法可以改成其他任何算法
        - The low-level is given an agent, ai, and a set of associated constraints. 
        It performs a search in the underlying graph to find an optimal path for agent 
        ai that satisfy all its constraints. Agent ai is solved in a decoupled manner, 
        i.e., while ignoring the other agents.
        """
        start = self.starts[agent]
        goal = self.dests[agent]

        open = []
        heapq.heappush(open, (0, 0, start, 0, [start]))  # (f, g, position, time, path)

        visited = {}

        index = 0
        while open:
            index += 1
            f, g, current, time, path = heapq.heappop(open)

            if current == goal:
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

                h = abs(move[0] - goal[0]) + abs(move[1] - goal[1])  # heuristic, 曼哈顿距离
                new_f = new_g + h

                state_key = (move, next_time)
                if state_key in visited and visited[state_key] <= new_g:
                    continue

                visited[state_key] = new_g
                heapq.heappush(open, (new_f, new_g, move, next_time, new_path))
        return None

    def validate(self, solution: dict[int, list[Point]]) -> list[tuple]:
        """
        校验所有的路径里面是否包含冲突, 返回所有的冲突的位置和时间点
        - A conflict is a tuple (ai, aj, v, t) where agent ai and agent aj occupy vertex v at time point t.

        注: 这里为了简化，我们只考虑顶点冲突(Vertex Conflict), 即，两个角色不能站在一个顶点上
        但是现实应用里面, 我们应该还要考虑:
            两个角色不能互相穿过对方(Swapping Conflict)
            第一个角色的下一个位置不能是第二个角色的起始位置(Following Conflict)
        """
        all_conflicts = []
        max_time = max(len(path) for path in solution.values())
        for t in range(max_time):
            occupied = {}   # 在时间点t时, 被占领的格子

            for agent, path in solution.items():
                if t < len(path):
                    pos = path[t]   # 获取 t 时间点时, agent 所处的位置
                    if pos in occupied:  # 如果所处的位置被其他 agent 所占领, 那么返回冲突
                        agent_j = occupied[pos]  # 与之冲突的 agent 索引
                        all_conflicts.append((agent, agent_j, pos, t))
                    occupied[pos] = agent

        return all_conflicts

    def visualize(self, solution: Optional[dict[int, list[Point]]]):
        """
        Deepseek 写的格式化输出
        """
        if not solution:
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
                t: int = len(path) - 1 if t >= len(path) else t
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
    ], dtype=int)
    starts = [(0, 0), (0, 4)]
    goals = [(0, 4), (0, 0)]

    cbs = CBS(grid, starts, goals, "4way")
    solution = cbs.high_level_search()
    cbs.visualize(solution)

    grid = numpy.array([
        [0, 0, 0, 0, -1],
        [0, -1, -1, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, 0, -1, -1],
        [0, -1, 0, 0, 0]
    ], dtype=int)
    starts = [(0, 0), (4, 4)]
    goals = [(4, 4), (0, 0)]

    cbs = CBS(grid, starts, goals, "4way")
    solution = cbs.high_level_search()
    cbs.visualize(solution)
