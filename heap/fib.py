"""
斐波那契堆, 仅包含插入和pop min两个功能
基于 chatgpt 写的代码, 个人做了一些bug修复和注释
"""

from typing import Optional, Self


class Node:
    def __init__(self, key: int | float) -> None:
        self.key: int | float = key
        self.degree: int = 0
        self.left: Self = self
        self.right: Self = self
        self.mark: bool = False
        """decrease-key相关标记, 如果只是插入和pop的话, 用不到mark这个参数"""
        self.parent: Optional[Self] = None
        self.child: Optional[Self] = None

    def sibs(self) -> list[Self]:
        """
        获得与自己同处于一颗树上的, 同级的所有节点
        """
        results = []

        node = self
        while True:
            results.append(node)
            node = node.right
            if node is self:
                break
        return results


class FibonacciHeap:
    def __init__(self):
        self.min: Optional[Node] = None
        """最小的元素所在的节点, 同时也可以判断 heap 是否为空 (如果是None就代表heap是空的)"""
        self.total_nodes = 0

    def add_right(self, node: Node, target: Node):
        """
        在 node 节点的右侧加入 target 节点
        [l] --- [node] --- [r]
        变成:
        [l] --- [node] --- [target] --- [r]
        """
        target.left = node
        target.right = node.right
        node.right.left = target
        node.right = target

    def remove_from_list(self, node: Node):
        """
        将 node 从 dl list 里面移除
        [l] --- [node] --- [r]
        变成:
        [l] --- [r]
        """
        node.left.right = node.right
        node.right.left = node.left
        node.left = node.right = node

    def insert(self, key: int | float):
        """
        插入一个新的元素, 插入的方式为,
        直接将元素作为一个单独的树, 然后将这个树直接加入到heap里面
        """
        node = Node(key=key)
        if self.min is None:
            self.min = node
        else:
            self.add_right(self.min, node)
            if node.key < self.min.key:
                self.min = node
        self.total_nodes += 1

    def get_min(self) -> Optional[int | float]:
        """获取最小值, 但不移除元素"""
        if self.min is None:
            return None
        return self.min.key

    def pop_min(self) -> Optional[int | float]:
        """
        移除并返回最小的那个值
        """
        z = self.min
        if z is None:
            return None

        # 将 z 的所有子节点都作为一个单独的树, 并将其加入到 root list 里面
        if z.child is not None:
            children = z.child.sibs()
            for child in children:
                # 重置 parent 和同级的所有其他节点
                child.parent = None
                child.left = child.right = child

                # 如果当前的min是空的, 那么直接设置 child 作为 min
                # 否则, 把 child 加入到 root list 中
                if self.min is None:
                    self.min = child
                else:
                    self.add_right(self.min, child)
            # z.child = None

        # 如果最顶端只有一个节点, 那么移除了这个节点之后 heap 就是空的
        if z.right == z:
            self.min = None

        # 否则, 移除 z 之后, 合并所有的树
        else:
            temp = z.right
            self.remove_from_list(z)
            self.min = temp
            self.consolidate()

        self.total_nodes -= 1
        return z.key

    def consolidate(self):
        """
        合并树, 使得heap里面不存在degree相同的两个树
        """
        if self.min is None:
            return

        A = dict()
        roots = self.min.sibs()
        for w in roots:
            x = w
            d = x.degree
            while d in A:
                y = A.pop(d)
                # 让更小的那个节点 y 成为 x 的子节点
                self.link(y if y.key > x.key else x,
                          x if y.key > x.key else y)
                d = x.degree
            A[d] = x

        # 重建 root list
        self.min = None
        for node in A.values():
            node.left = node.right = node
            if self.min is None:
                self.min = node
            else:
                node.left = self.min
                node.right = self.min.right
                self.min.right.left = node
                self.min.right = node
                if node.key < self.min.key:
                    self.min = node

    def link(self, y: Node, x: Node):
        """
        合并两个树, 让节点 y 成为节点 x 的子节点
        """
        self.remove_from_list(y)
        y.parent = x

        if x.child is None:
            x.child = y
            y.left = y.right = y
        else:
            self.add_right(x.child, y)
        x.degree += 1
        y.mark = False

    def print_heap(self):
        """
        AI写的打印当前所有的树的代码
        """
        print("=" * 25)
        if self.min is None:
            print("堆为空")
            return

        print(f"最小值: {self.min.key})")

        # 打印出所有的树
        roots = self.min.sibs()
        for i, root in enumerate(roots):
            print(f"树 {i + 1}:")
            self._print_tree(root)
        print()

    def _print_tree(self, root: Node, prefix: str = "", is_last: bool = True):
        # 当前节点的标记
        marker = "└── " if is_last else "├── "

        # 打印当前节点
        if root.parent is None:
            print(prefix + marker + f"[r: {root.key}]")
        else:
            # 找到当前节点在父节点孩子中的位置
            parent_children = root.parent.child.sibs()
            child_index = parent_children.index(root) + 1
            print(prefix + marker + f"[{self._get_parent_path(root.parent)}-{child_index}: {root.key}]")

        # 更新前缀
        new_prefix = prefix + ("    " if is_last else "│   ")

        # 递归打印所有子节点
        if root.child is not None:
            children = root.child.sibs()
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                self._print_tree(child, new_prefix, is_last_child)

    def _get_parent_path(self, node: Node) -> str:
        """
        获取节点的完整路径标记
        """
        if node.parent is None:
            return "r"  # root
        else:
            # 找到当前节点在父节点孩子中的位置
            parent_children = node.parent.child.sibs()
            child_index = parent_children.index(node) + 1
            return f"{self._get_parent_path(node.parent)}-{child_index}"


if __name__ == "__main__":
    h = FibonacciHeap()
    for k in reversed(range(12)):
        h.insert(k)

    h.print_heap()
    print("="*20)
    for _ in range(10):
        h.pop_min()
        h.print_heap()
