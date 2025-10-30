"""
Horspool算法, 也称boyer-moore-horspool算法
属于boyer-moore算法的简化版本
"""


def build_table(pattern: str) -> dict[str, int]:
    """
    构建Horspool的跳转表
    对于字符串[pattern]中的每个字符（除最后一个），记录它到末尾的距离
    """
    m = len(pattern)
    shift_table = {}

    # 对于模式串中的前m-1个字符
    for i in range(m - 1):
        # 计算该字符到模式串末尾的距离
        shift_table[pattern[i]] = m - 1 - i

    return shift_table


def horspool_search(text: str, pattern: str):
    """
    查找所有匹配位置
    """
    positions = []
    n = len(text)
    m = len(pattern)

    if m == 0 or m > n:
        return positions

    shift_table = build_table(pattern)

    i = m - 1
    while True:
        if i > n:
            break

        # 过程可视化
        print(text)
        alignment = "." * (i - m + 1) + pattern  # 用字符.填充对齐
        print(alignment)
        print(f"当前位置: i = {i}, 字符 = '{text[i]}'")

        k = 0
        while k < m:
            t_char = text[i - k]
            p_char = pattern[m - 1 - k]
            if t_char != p_char:
                break
            k += 1

        if k == m:
            positions.append(i - m + 1)
            i += 1  # 移动到下一个位置继续搜索
        else:
            # 查看最后一个字符所对应的表
            # 注意, 是最后一个字符, 而非匹配处的字符
            # 反例: 如果是匹配处的字符的话, 就会出现
            # text = "abcababaaa"
            # pattern = "cabab"
            # 这类错误的现象
            char = text[i]
            print(text[:i], "[", text[i], "]", text[i+1:])
            s = shift_table[char] if char in shift_table else m
            i += s
        print("\n\n")

    return positions


if __name__ == "__main__":
    text = "abcdeabbbdabcab"
    pattern = "bcab"
    print(f"文本: {text}")
    print(f"模式: {pattern}")

    all_positions = horspool_search(text, pattern)
    print(f"所有位置: {all_positions}")
