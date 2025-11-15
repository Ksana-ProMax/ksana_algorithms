"""
Meta-agent Conflict-Based Search For Optimal Multi-Agent Path Finding
Sharon,  G.;  Stern,  R.;  Felner,  A.;  and  Sturtevant,  N.  R.

基于CBS的优化版本
注释里面，列表形式的注释，为论文原文
"""


import copy
from dataclasses import dataclass
import heapq
from typing import Optional, Self
from itertools import combinations
import numpy
from cbs import Constraint, Point
from cbs_multi import CBSMulti


@dataclass
class MetaConstraint:
    """
    - meta-constraint for a given meta-agent x is a tuple (x; v; t) 
    where any individual agent xi in x is prohibited from occupying vertex v at time step t.
    """
    meta_id: int
    """meta agent的索引"""
    coord: Point
    """不能出现的地点"""
    time: int
    """不能出现的时间点t"""


@dataclass
class MetaAgent:
    """
    合并的agents, Meta Agent只会不断合并, 不会做拆分

    - A meta-agent consists of M agents, each agent is associated with its
    own position. A single agent is just a meta-agent of size 1
    - The lowlevel search for a meta-agent of size M is in fact an optimal MAPF
    problem for M agents, and is solved with a coupled MAPF solver
    """
    id: int
    agents: list[int]


@dataclass
class Node:
    constraints: list[MetaConstraint]
    """路径约束列表"""
    cost: int
    """所有agent的总成本"""
    solution: dict[int, list[Point]]
    """每个agent的路径"""
    meta_agents: dict[int, MetaAgent]
    """meta id到具体meta agent的映射表"""
    goal: bool
    parent: Optional[Self] = None

    def __lt__(self, other):
        """heap使用时的比较函数"""
        return self.cost < other.cost


class MACBS:
    def __init__(self, grid: numpy.ndarray,
                 starts: list[Point],
                 dests: list[Point],
                 bound: int = 1,
                 move_type: str = "4way") -> None:
        self.grid: numpy.ndarray = grid
        self.rows, self.cols = grid.shape
        assert self.rows > 0 and self.cols > 0

        self.starts: list[Point] = starts
        self.dests: list[Point] = dests
        assert self.starts and self.dests
        self.num_agents = len(starts)

        self.bound: int = bound
        """
        - In our merging policy we identify when agents should be merged using a bound parameter, B. 
        Two agents a_i; a_j are merged into a meta-agent a_i,j if the number of conflicts between a_i and a_j 
        seen so far during the search exceeds B.
        """
        self.conflict_matrix: dict[tuple, int] = {}
        """
        - CM[i; j] stores the number of conflicts between agents a_i and a_j seen so far by MA-CBS. 
         After a new conflict between a_i and a_j is found (Line 10) CM[i; j] is incremented by 1.
        """

        # 移动方向
        self.move_type = move_type
        if move_type == "4way":
            # 上、下、左、右, 以及允许原地停留
            self.dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            # 上、下、左、右, 以及对角线，以及允许原地停留
            self.dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def high_level_search(self):
        """
        论文里面 Algorithm 1 的算法流程

        - First, MA-CBS requires a merging policy to decide which option to choose (branch or merge) 
        (Line 11). Second, MA-CBS requires a mechanism to define the constraints imposed on the new 
        meta-agent (Line 13).
        """
        open = []

        # 最初始的阶段，每个agent都是独立的meta agent
        meta_agents: dict[int, MetaAgent] = {}
        for id in range(self.num_agents):
            meta_agent = MetaAgent(id=id, agents=[id])
            meta_agents[id] = meta_agent

        meta_agent_ids = list(range(self.num_agents))
        for i, j in combinations(meta_agent_ids, 2):
            self.conflict_matrix[(i, j)] = 0

        solution = {}
        cost = 0
        for id in range(self.num_agents):
            meta_agent: MetaAgent = meta_agents[id]
            paths = self.low_level_search(meta_agent, [])
            if not paths:
                return None  # 无解
            solution.update(paths)
            cost += sum(len(path) - 1 for path in paths.values())

        # 将 root 节点加入到 heap 上面
        root = Node(constraints=[], cost=cost, solution=solution, meta_agents=meta_agents, goal=False)
        heapq.heappush(open, root)

        while open:
            current_node: Node = heapq.heappop(open)
            meta_conflict = self.validate(current_node)

            # 如果没有冲突, 那么直接返回最优结果即可
            if not meta_conflict:
                current_node.goal = True
                return current_node.solution
            meta_x_id, meta_y_id, coord, time, agent_i, agent_j = meta_conflict

            # 更新冲突矩阵, 并根据矩阵的值判断是否需要合并
            # 如果需要合并, 那么合并两个 meta agent 以及对应的 meta constraint
            key = (min(agent_i, agent_j), max(agent_i, agent_j))
            self.conflict_matrix[key] += 1
            should_merge = self.conflict_matrix.get(key, 0) > self.bound

            if should_merge:
                merged_node = self.merge_meta_agents(current_node, meta_x_id, meta_y_id)
                if merged_node:
                    heapq.heappush(open, merged_node)
            else:
                for constrained_meta_id in [meta_x_id, meta_y_id]:
                    new_constraints = current_node.constraints + [
                        MetaConstraint(constrained_meta_id, coord, time)
                    ]

                    new_solution = copy.deepcopy(current_node.solution)
                    new_meta_agents = copy.deepcopy(current_node.meta_agents)

                    constrained_meta_agent = new_meta_agents[constrained_meta_id]
                    new_paths = self.low_level_search(constrained_meta_agent, new_constraints)
                    if new_paths:
                        new_solution.update(new_paths)
                        new_cost = sum(len(path) - 1 for path in new_solution.values())

                        new_node = Node(
                            constraints=new_constraints,
                            cost=new_cost,
                            solution=new_solution,
                            meta_agents=new_meta_agents,
                            goal=False,
                            parent=current_node
                        )
                        heapq.heappush(open, new_node)

    def low_level_search(self,
                         meta_agent: MetaAgent,
                         meta_constraints: list[MetaConstraint]) -> Optional[dict[int, list[Point]]]:
        """
        由于 joint search 不限制搜索算法, 
        所以使用 CBSMulti 进行 joint search, CBSMulti 支持一个及以上 agent 的搜索
        - Recall again that MA-CBS(0) is equivalent to A*+ID (or any other MAPF 
        solver used instead of A* for the low-level search)
        """

        agent_ids = list(meta_agent.agents)
        starts = [self.starts[agent_id] for agent_id in agent_ids]
        dests = [self.dests[agent_id] for agent_id in agent_ids]

        # 将 meta 约束转换为 CBS 约束
        cbs_constraints = []
        for constraint in meta_constraints:
            # 找到 meta agent 对应的 meta constraint,
            # 然后将其施加在 meta agent 之下的所有 agent 上
            if constraint.meta_id == meta_agent.id:
                for agent_id in agent_ids:
                    cbs_constraints.append(Constraint(agent_id, constraint.coord, constraint.time))

        # 利用 CBS 求解
        cbs_solver = CBSMulti(self.grid, starts, dests, move_type=self.move_type)
        solution = cbs_solver.high_level_search()

        # 将 CBS 解的 id 映射回原始的agent ID
        if solution:
            mapped_solution = {}
            for i, agent_id in enumerate(agent_ids):
                mapped_solution[agent_id] = solution[i]
            return mapped_solution

        return None

    def validate(self, node: Node) -> Optional[tuple[int, int, Point, int, int, int]]:
        """
        检测是否存在冲突
        同样的, 这里为了简化, 只考虑顶点冲突 (Vertex Conflict)

        Returns:
            meta agent x 的 id, meta agent y 的 id, 冲突的地点, 发生冲突的 agent i 的 id, 发生冲突的 agent j 的 id
        """
        solution = node.solution
        max_time = max(len(path) for path in solution.values()) if solution else 0

        for t in range(max_time):
            position_agents: dict[Point, list] = {}

            # 遍历agent的路径
            for agent_id, path in solution.items():
                if t < len(path):
                    pos = path[t]
                    if pos not in position_agents:
                        position_agents[pos] = []
                    position_agents[pos].append(agent_id)

            for pos, agents in position_agents.items():
                if len(agents) > 1:
                    agent_i, agent_j = agents[0], agents[1]

                    meta_x_id: Optional[int] = None
                    meta_y_id: Optional[int] = None

                    # 找到 agent 所属的 meta agent
                    for meta_id, meta_agent in node.meta_agents.items():
                        if agent_i in meta_agent.agents:
                            meta_x_id = meta_id
                        if agent_j in meta_agent.agents:
                            meta_y_id = meta_id

                    # 只有在两个不同的 meta agent 产生路径冲突的时候, 才返回对应的冲突
                    if meta_x_id is not None and meta_y_id is not None and meta_x_id != meta_y_id:
                        return (meta_x_id, meta_y_id, pos, t, agent_i, agent_j)

        return None

    def merge_meta_agents(self, node: Node, meta_x_id: int, meta_y_id: int) -> Optional[Node]:
        """
        合并两个meta agent

        - The merging process is performed as follows. Assume a CT node N with k agents. 
        Suppose that agents a1, a2 were chosen to be merged. We now have k-1 agents with a 
        new meta-agent of size 2, labeled a1;2. This meta-agent will never be split again in 
        the subtree of the CT below this given node; it might, however, be merged with other 
        (meta) agents to new meta-agents.
        """
        meta_x: MetaAgent = node.meta_agents[meta_x_id]
        meta_y: MetaAgent = node.meta_agents[meta_y_id]

        # 创建一个新的 meta agent, 以更小的那个 id 作为合并后的新 id
        # 然后, 合并两个 meta 里面的所有 agent
        combined_meta_id: int = min(meta_x_id, meta_y_id)
        combined_agent_ids: list[int] = meta_x.agents + meta_y.agents
        combined_meta_agent = MetaAgent(id=combined_meta_id, agents=combined_agent_ids)

        # 更新 meta agents 映射
        node_meta_agents = copy.deepcopy(node.meta_agents)
        node_meta_agents.pop(meta_x_id)
        node_meta_agents.pop(meta_y_id)
        node_meta_agents[combined_meta_id] = combined_meta_agent

        # 合并约束
        combined_constraints = self.merge_constraints(node.constraints, meta_x_id, meta_y_id, combined_meta_id)

        # 为新的 meta agent 寻找路径, 然后更新 solution
        new_solution = copy.deepcopy(node.solution)
        new_paths = self.low_level_search(combined_meta_agent, combined_constraints)
        if not new_paths:
            return None

        # 更新 solution 和 cost
        for agent_id in combined_agent_ids:
            if agent_id in new_solution:
                new_solution.pop(agent_id)
        new_solution.update(new_paths)
        new_cost = sum(len(path) - 1 for path in new_solution.values())

        return Node(
            constraints=combined_constraints,
            cost=new_cost, solution=new_solution,
            meta_agents=node_meta_agents, goal=False, parent=node
        )

    def merge_constraints(self, constraints: list[MetaConstraint],
                          meta_x_id: int,
                          meta_y_id: int,
                          new_meta_id: int) -> list[MetaConstraint]:
        """
        合并两个meta agent的约束

        - These conflicts (and therefore the resulting constraints) can be divided to three groups.
        (1) internal: conflicts between ai and aj .
        (2) external(i): conflicts between a_i and any other agent a_k (where k != j).
        (3) external(j): conflicts between a_j and any other agent a_k (where k != i).
        - For each external constraint (ai; v; t) we add a meta constraint (ai;j ; v; t). 
        Similarly, for each external constraint (aj ; v; t) we add a meta constraint (ai;j ; v; t).
        - internal conflicts are no longer relevant as ai and aj will be solved in a coupled manner by the low-level solver.
        """
        combined_constraints = []

        for constraint in constraints:
            # 和这两个 agent 其中一方有关的 constraint
            # 将 constraint 的 id 修改为新的 meta_agent 的 id
            if (constraint.meta_id == meta_x_id) or (constraint.meta_id == meta_y_id):
                new_constraint = MetaConstraint(new_meta_id, constraint.coord, constraint.time)
                combined_constraints.append(new_constraint)
            else:
                # 和这两个 agent 无关的其他的 constraint
                combined_constraints.append(constraint)

        return combined_constraints

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

    starts = [(0, 0), (0, 4), (2, 4), (2, 0), (0, 2), (2, 2)]
    dests = [(0, 4), (0, 0), (2, 0), (2, 4), (2, 2), (0, 2)]

    macbs = MACBS(grid, starts, dests, bound=3, move_type="4way")
    solution = macbs.high_level_search()
    macbs.visualize(solution)
