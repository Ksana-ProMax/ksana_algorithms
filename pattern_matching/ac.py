"""
Aho Corasick自动机算法
推荐参考: https://cp-algorithms.com/string/aho_corasick.html
"""

from dataclasses import dataclass
from typing import Optional, Self


@dataclass
class Node:
    """代表了trie上面的一个节点"""
    name: str
    outputs: list[str]  # 代表找到了匹配
    children: dict[str, Self]   # 子节点
    link: Optional[Self] = None  # fail-link

    def __repr__(self) -> str:
        return f"名称: {self.name}\n子节点: {[c.name for _, c in self.children.items()]} Link: {self.link.name if self.link is not None else "无"} Out: {self.outputs}"


class AhoCorasick:
    def add_patterns(self, patterns: list[str]):
        index = 0
        for pattern in patterns:
            node = self.root
            for idx, char in enumerate(pattern):
                # children 里面如果没有 char 这个字符, 那么新构造一个 child
                if char not in node.children:
                    node.children[char] = Node(f"<ID: {index}>-{pattern[:idx]}[{char}]", [], {})
                    index += 1
                node = node.children[char]
            node.outputs.append(pattern)

    def build_failure_links(self):
        """
        建立failure link, 
        failure link 的意思是, 如果我们在某个分支的某个节点匹配失败了
        那么我们需要找到其他的分支，这个分支的条件是，prefix等于当前分支的suffix

        Example:
            - 如果当前分支是 a-b-c-d, 在匹配到 d 这个节点匹配失败了, 
            那么 a-b-[c] 的这个地方, 可以跳转到其他的分支上, 例如 b-c-[b]-a
        """
        queue: list[Node] = [self.root]

        while True:
            if len(queue) == 0:
                break

            cur_node: Node = queue.pop(0)

            for char, child_node in cur_node.children.items():
                cur_fail = cur_node.link

                # 沿着 fail link 往上追溯, 直至 link 的 children 里面包含 char 这个字符为止
                # 然后 child_node 和 link.children[char] 这个节点建立连接
                while (cur_fail is not None) and (char not in cur_fail.children):
                    cur_fail = cur_fail.link

                if cur_fail is not None and char in cur_fail.children:
                    child_node.link = cur_fail.children[char]
                else:
                    child_node.link = self.root

                if child_node.link is not None:
                    # 如果字符串 A 包含了字符串 B (比方说 abcd 和 bc),
                    # 那么指针在 A 分支的时候, 应该具备匹配 B 字符串的能力
                    # 所以, 我们从节点沿着 fail link 追溯的时候, 路径上的所有节点都应该包含 B
                    child_node.outputs += child_node.link.outputs
                queue.append(child_node)

    def build_trie(self, patterns: list[str]):
        """
        构造trie树状结构, 以及构造failure link
        """
        self.root = Node("root", [], {})
        self.add_patterns(patterns)
        self.build_failure_links()

        # print所有结点的信息
        queue: list[Node] = [self.root]
        while True:
            if len(queue) == 0:
                break

            cur_node = queue.pop(0)
            print(cur_node, "\n")
            for _, child in cur_node.children.items():
                queue.append(child)

    def ac_search(self, text: str) -> list[tuple[str, int, int]]:
        node = self.root
        results = []
        for i, char in enumerate(text):
            # 沿着 trie 的线路移动, 如果 char 属于当前 node children 的一员
            # 那么不需要进入 while 循环
            while (node is not None) and (char not in node.children):
                node = node.link

            if node is not None and char in node.children:
                node = node.children[char]
            else:
                node = self.root

            if len(node.outputs) > 0:   # 找到了匹配
                for pat in node.outputs:
                    start = i - len(pat) + 1
                    results.append((pat, start, i + 1))
        return results


if __name__ == "__main__":
    text = "abbbcab"
    patterns = ["aabb", "abbc", "cab", "bcab"]
    ac = AhoCorasick()
    ac.build_trie(patterns)
    matches = ac.ac_search(text)

    print(f"文本 {text}")
    print(f"模式 {patterns}")

    for pat, s, e in matches:
        print(f"找到匹配: {pat} 位于索引 {s}")
