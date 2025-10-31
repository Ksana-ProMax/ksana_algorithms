"""
Rabin–Karp算法
利用哈希值进行快速匹配
代码取自: http://www-igm.univ-mlv.fr/~lecroq/string/node5.html#SECTION0050
"""


def rehash(front: int, back: int, hash: int, d: int):
    """
    哈希值重新计算, 根据当前哈希值 - 第一个字符 + 最后一个字符
    来计算新的哈希值

    Args:
        - front : 窗口第一个字符
        - back : 窗口最后一个字符
        - hash: 当前哈希值
        - d : 2**n
    """
    # 参考代码里面, 对哈希值进行了mod运算, 这里把这个运算去掉了
    return (hash - front * d) * 2 + back


def rk_search(text: str, pattern: str):
    len_text = len(text)
    len_pattern = len(pattern)

    if len_pattern == 0 or len_text == 0 or len_pattern > len_text:
        return []

    # 计算初始哈希值
    h_pattern = 0
    h_text = 0
    for i in range(len_pattern):
        h_pattern = h_pattern * 2 + ord(pattern[i])
        h_text = h_text * 2 + ord(text[i])

    j = 0
    d = 2 ** (len_pattern-1)
    results = []
    while True:
        if j >= len_text - len_pattern + 1:
            break   # 之后的位置无需匹配

        # 如果哈希值匹配, 进行精确比较, 验证每一个对应的字符是否匹配
        if h_pattern == h_text:
            if pattern == text[j:j + len_pattern]:
                results.append(j)

        if j < len_text - len_pattern:
            h_text = rehash(ord(text[j]), ord(text[j + len_pattern]), h_text, d)

        j += 1
    return results


# 测试示例
if __name__ == "__main__":
    # 测试数据
    text = "ABCDEF"
    pattern = "CDEF"

    # 使用基本版本
    results = rk_search(text, pattern)
    print(f"匹配位置: {results}")
