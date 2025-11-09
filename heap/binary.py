"""
Binary Heap
Python heapq 的源代码照搬过来的, 整体思路是不管怎么样先子节点上浮，然后再下移
加了一个Deepseek写的可视化函数
"""


import math


def print_heap_tree(heap: list):
    """Deepseek写的可视化函数"""
    if heap is None:
        return

    n = len(heap)
    height = math.floor(math.log2(n)) + 1 if n > 0 else 0
    max_width = 2 ** (height - 1) * 4 - 1

    current_index = 0
    for level in range(height):
        level_nodes = 2 ** level
        level_end = min(current_index + level_nodes, n)

        # Calculate spacing
        spacing_between = max_width // (2 ** level)
        spacing_before = (spacing_between - 1) // 2

        # Build the level string
        level_str = " " * spacing_before
        for i in range(current_index, level_end):
            level_str += f"{heap[i]:2d}"
            if i < level_end - 1:
                level_str += " " * spacing_between
            else:
                level_str += " " * spacing_before

        print(level_str.center(max_width))
        current_index = level_end
    print(heap)
    print()


def _siftdown(heap: list, startpos: int, pos: int):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup(heap: list, pos: int):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    childpos = 2*pos + 1    # 左子节点的位置
    while childpos < endpos:
        rightpos = childpos + 1
        # 将childpos设为较小的子节点
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # 将较小的子节点上移
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


def heapify(x: list):
    n = len(x)
    for i in reversed(range(n//2)):
        _siftup(x, i)
        print_heap_tree(x)


def heappop(heap: list):
    lastelt = heap.pop()
    if len(heap) > 0:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)

        return returnitem
    return lastelt


if __name__ == "__main__":
    ls = [5, 7, 4, 10, 9, 3, 6]
    print_heap_tree(ls)
    heapify(ls)

    heappop(ls)
    print_heap_tree(ls)
