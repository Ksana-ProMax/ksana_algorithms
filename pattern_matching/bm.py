"""
Boyer-Moore算法
在horspool算法的基础上加入good suffix表格
建议参考: https://www.geeksforgeeks.org/dsa/boyer-moore-algorithm-good-suffix-heuristic/
论文出处:
A Fast String Searching Algorithm 
Robert S. Boyer, J Strother Moore
"""


def build_delta1_table(pattern: str) -> dict[str, int]:
    """
    构建Horspool的跳转表
    对于字符串[pattern]中的每个字符（除最后一个），记录它到末尾的距离
    """
    m = len(pattern)
    table = {}

    # 对于模式串中的前m-1个字符
    for i in range(m - 1):
        # 计算该字符到模式串末尾的距离
        table[pattern[i]] = m - 1 - i

    return table


def build_delta2_table(pattern: str) -> list[int]:
    """
    构建good suffix表 (delta 2)
    deepseek写的, 应该是正确的
    表内的元素为不匹配发生时前进的距离

    Example:
        匹配的后3个元素为 ?ABC, 那么需要在 pattern 找到除了匹配以外, 
        其他的 *ABC 的地方，并且 *ABC 和 ?ABC, *的元素需要和?的元素不一样
        然后让 ABC 与文本的 ABC 对齐
    """
    len_pattern = len(pattern)

    table: list[int] = [0 for _ in range(len_pattern+1)]
    fail: list[int] = [0 for _ in range(len_pattern)] + [len_pattern + 1]

    i = len_pattern
    j = len_pattern + 1

    while i > 0:
        while j <= len_pattern:
            p_im1 = pattern[i - 1]
            p_jm1 = pattern[j - 1]
            if p_im1 == p_jm1:
                break

            if table[j] == 0:
                table[j] = j - i
            j = fail[j]
        i -= 1
        j -= 1
        fail[i] = j

    j = fail[0]
    for i in range(len_pattern + 1):
        if table[i] == 0:
            table[i] = j
        if i == j:
            j = fail[j]

    return table


def bm_search(text: str, pattern: str):
    """
    查找所有匹配位置
    """
    positions = []
    len_text = len(text)
    len_pattern = len(pattern)

    if len_pattern == 0 or len_pattern > len_text:
        return positions

    print(f"构造两个表格: ")
    delta_1_table = build_delta1_table(pattern)
    delta_2_table = build_delta2_table(pattern)
    print(f"表1: {delta_1_table}\n表2: {delta_2_table}")

    i = 0       # i 代表了文本的指针
    while True:
        if i > len_text-len_pattern:
            print("匹配结束")
            break

        # 过程可视化
        print(text)
        print("." * (i - len_pattern + 1) + pattern)  # 用字符.填充对齐

        j = len_pattern - 1     # j 代表了模式的指针
        while j >= 0:
            t_char = text[i+j]
            p_char = pattern[j]
            if t_char != p_char:
                break
            j -= 1

        if j < 0:
            positions.append(i)
            i += delta_2_table[0]  # 根据 delta 2 表格, 移动到下一个位置继续搜索
        else:
            char = text[i]
            print(text[:i], "[", text[i], "]", text[i+1:])

            # 比较, 取较大的移动距离
            s_from_delta1 = delta_1_table[char] if char in delta_1_table else len_pattern
            s_from_delta2 = delta_2_table[j + 1]
            s = max(s_from_delta1, s_from_delta2)
            print(f"根据表1, 前进{s_from_delta1}, 根据表2, 前进{s_from_delta2}, 最终: 前进{s}")

            i += s
        print("\n")

    return positions


if __name__ == "__main__":
    text = "aaabaaaaabaa"
    pattern = "aabaa"
    print(f"文本: {text}")
    print(f"模式: {pattern}")

    all_positions = bm_search(text, pattern)
    print(f"所有位置: {all_positions}")
