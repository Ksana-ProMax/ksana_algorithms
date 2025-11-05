"""
Suffix Automation 简化版本
从起点出发到终点, 路径中的字符构成了字符串的后缀
推荐参考: https://cp-algorithms.com/string/suffix-automaton.html
"""

from copy import deepcopy
from dataclasses import dataclass


@dataclass
class State:
    name: str
    length: int                 # 计数用, 用于比较 len(p)+1 和 len(q)
    next: dict[str, int]        # 跳转表
    link: int                   # suffix link
    is_terminal: bool = False   # 是否为终止节点

    def __repr__(self) -> str:
        return f"Name: {self.name}\nLength: {self.length} Link: {self.link} Next: {self.next} T: {self.is_terminal}"


class SuffixAutomaton:
    def __init__(self):
        initial_state = State(name="init", length=0, link=-1, next={})   # 初始节点/状态
        self.states: list[State] = [initial_state]
        self.last: int = 0  # 代表了字符 x 之前的所有字符组成的前缀的节点/状态

    def extand_char(self, char: str) -> None:
        """
        在目前的自动机中，加入字符 c 构成新的自动机
        """
        assert len(char) == 1   # 一次性只能扩展一个字符

        # 每次有一个新的字符，创建一个新的 state
        q_copy = State(name=f"Index: {len(self.states)}", length=self.states[self.last].length + 1, link=0, next={})
        self.states.append(q_copy)

        cur: int = len(self.states) - 1
        p: int = self.last

        while p != -1:
            if char in self.states[p].next:
                break

            # 建立指向链接, 然后追溯到 last 的上一级, last 的上上一级……
            # 直至 char 处于 states[p] 的指向路径中为止
            self.states[p].next[char] = cur
            p = self.states[p].link

        # 找不到图中字符c的指向/链接, 那么当前节点的suffix link将指向初始节点
        if p == -1:
            self.states[cur].link = 0
        else:
            q: int = self.states[p].next[char]
            # 如果长度只差1, 那么可以直接建立连接
            if self.states[p].length + 1 == self.states[q].length:
                self.states[cur].link = q
            else:
                # 复制一个状态 Q
                q_copy = State(name=f"Index: {len(self.states)}", length=self.states[p].length + 1, link=self.states[q].link, next={})
                self.states.append(q_copy)
                new_q_index = len(self.states) - 1

                # 复制 next
                q_copy.next = deepcopy(self.states[q].next)
                state_p = self.states[p]
                while p != -1 and state_p.next.get(char) == q:
                    state_p.next[char] = new_q_index    # 修改被复制节点相关的跳转
                    state_p = self.states[state_p.link]

                # 重新设置 suffix link
                self.states[q].link = new_q_index
                self.states[cur].link = new_q_index

        self.last = cur

    def set_terminal_states(self):
        """设置所有的terminal节点"""
        p = self.last
        while p != -1:
            self.states[p].is_terminal = True
            p = self.states[p].link

    def build(self, s: str) -> None:
        """构造自动机"""
        for ch in s:
            self.extand_char(ch)
        self.set_terminal_states()

        # 输出所有节点的信息
        for st in self.states:
            print(st, "\n")

    def contains(self, pattern: str) -> bool:
        """
        通过自动机检测是否包含了 pattern
        """
        v = 0
        for char in pattern:
            if char in self.states[v].next:
                v = self.states[v].next[char]
            else:
                return False
        return True


if __name__ == "__main__":
    s = "abcabb"
    sam = SuffixAutomaton()
    sam.build(s)

    pattern = "abb"
    print(f"是否有匹配: {sam.contains(pattern)}")
