"""
Bitap算法
利用bitwise算法来找到匹配的字符串
"""


def build_dict_B(text: str):
    """
    构造长度为 len(text) 的映射 B
    每个字符在 text 里面出现的位置(从右往左), 这个位置上的值为1

    Example:
        abcb, 
            - a 在位置0出现, 对应1000, 
            - b 在位置1,3出现, 对应0101
            - c 在位置2出现, 对应0010
            - 其他字符: 0000
    """
    B = {char: 0 for char in set(text)}
    for i, char in enumerate(text):
        # 从右往左，第i位设为1
        B[char] |= (1 << i)

    return B


def bitap_search(text: str, pattern: str) -> list:
    """搜索pattern, 返回匹配的地方的索引"""

    n = len(text)
    m = len(pattern)

    B = build_dict_B(pattern)

    # 备注:
    # "Review on String-Matching Algorithm", Zhaoyang Zhang
    # 这篇论文里面把初始值设置为了1, 虽然设置为1和0并不会影响结果, 但我认为这是一个错误
    D = 0

    # 用于获取最左侧的bit的mask
    # 将 ...001 进行 bitshift 左移 (m - 1) 次
    highest_bit_mask = 1 << (m - 1)

    results = []
    for i, char in enumerate(text):
        char_mask = B.get(char, 0)

        # 更新状态
        D = ((D << 1) | 1) & char_mask

        if D & highest_bit_mask != 0:    # (D&2^(m-1))≠0
            match_pos = i - m + 1
            results.append(match_pos)

    return results


if __name__ == "__main__":
    text = "abcabbcca"
    pattern = "bc"

    print(f"文本: {text}")
    print(f"模式: {pattern}")

    matches = bitap_search(text, pattern)
    print(f"匹配位置: {matches}")
