"""
Pairing Heap 配对堆

The Pairing Heap: A New Form of Self-Adjusting Heap
Michael L. Fredman, Robert Sedgewick, Daniel D. Sleator, and Robert E. Tarjan
"""

from dataclasses import dataclass
from typing import Optional, Self


@dataclass
class Node:
    key: int | float
    parent: Optional[Self] = None
    child: Optional[Self] = None
    sibling: Optional[Self] = None

    def children(self) -> list[Self]:
        """获取所有的子节点"""
        res = []
        cur = self.child
        while cur is not None:
            res.append(cur)
            cur = cur.sibling
        return res


class PairingHeap:
    def __init__(self) -> None:
        self.root: Optional[Node] = None

    def insert(self, key: int | float) -> None:
        """
        插入一个新的元素
        """
        new_node = Node(key)
        if self.root is None:
            self.root = new_node
        else:
            y = self.root if self.root.key > new_node.key else new_node
            x = new_node if self.root.key > new_node.key else self.root
            self.root = self.link(y, x)

    def link(self, y: Node, x: Node) -> Node:
        """
        让节点 y 成为节点 x 的子节点
        """
        y.parent = x
        y.sibling = x.child
        x.child = y
        return x

    def find_min(self) -> Optional[int | float]:
        """Return the minimum key without removing it, or None if empty."""
        if self.root is None:
            return None
        return self.root.key

    def meld(self, other: Self) -> None:
        """
        合并两个树（清空另外一颗树）
        """
        if other.root is None:
            return
        if self.root is None:
            self.root = other.root
        else:
            y = self.root if self.root.key > other.root.key else other.root
            x = other.root if self.root.key > other.root.key else self.root
            self.root = self.link(y, x)

        other.root = None

    def pop_min(self):
        if self.root is None:
            return None

        min = self.root.key
        children = self.root.children()
        if len(children) == 0:
            self.root = None
        else:
            # 截断所有与root之间的链接
            for child in children:
                child.parent = None

            paired = self.first_pass(children)
            self.root = self.second_pass(paired)
        return min

    def first_pass(self, children: list[Node]) -> list[Node]:
        """
        参考论文 Fig.6 和 Fig.7
        对所有的根节点进行两两配对, 让更小key的节点成为root节点
        """
        paired: list[Node] = []
        i = 0
        n = len(children)
        while i + 1 < n:
            a = children[i]
            b = children[i + 1]
            paired.append(self.link(b if a.key <= b.key else a, a if a.key <= b.key else b))
            i += 2

        # 存在多余的无法配对的节点, 则将其单独作为一个树
        if i < n:
            last = children[i]
            last.parent = None
            last.sibling = None
            paired.append(last)
        return paired

    def second_pass(self, paired: list[Node]) -> Optional[Node]:
        if len(paired) == 0:
            return None
        while len(paired) > 1:
            # list 的 pop 将弹出最后一个元素，所以合并的方式是从右到左
            a = paired.pop()
            b = paired.pop()
            if a.key <= b.key:
                paired.append(self.link(b, a))
            else:
                paired.append(self.link(a, b))
        return paired[0]


if __name__ == "__main__":
    h = PairingHeap()
    for k in range(8):
        h.insert(k)
    print("min:", h.find_min())

    for _ in range(5):
        h.pop_min()
        print("min:", h.find_min())
