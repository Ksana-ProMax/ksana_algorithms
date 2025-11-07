"""
Two-way string matching 的简化版本, 只有 large period 情况下的代码
另外一种情况参见原论文

MAXIME CROCHEMORE AND DOMINIQUE PERRIN
L.I. T. P., Institut Blaise Pascal, Universit6 Paris 7, 2, Place Jussieu, Paris Cedex 05, France

推荐参考: http://www-igm.univ-mlv.fr/~lecroq/string/node26.html
"""


def maximal_suffix(x: str, reverse=False):
    """
    论文第四节的 Maximal Suffixes, 流程为 Fig 17的算法
    Maximal Suffixes 为按照字母表顺序排序之后, 最大的后缀所在的位置和周期

    Args:
        - reverse : 是否倒转字母表的大小排序
    """
    n = len(x)
    i, j, k, p = 0, 1, 0, 1

    while j + k < n:
        a_prime = x[i + k] * (-1 if reverse else 1)
        a = x[j + k] * (-1 if reverse else 1)
        if a < a_prime:
            j = j + k + 1
            k = 0
            p = j - i
        elif a == a_prime:
            if k + 1 == p:
                j = j + p
                k = 0
            else:
                k += 1
        else:
            i = j
            j = i + 1
            k = 0
            p = 1
    return i, p


def cf(pattern: str):
    """
    寻找周期, 流程为 Fig 19的算法的前半部分
    """
    i1, p1 = maximal_suffix(pattern, reverse=False)
    i2, p2 = maximal_suffix(pattern, reverse=True)

    l = i1 if i1 >= i2 else i2
    p = p1 if i1 >= i2 else p2

    print(f"字符串 {pattern} 的正序: {(i1, p1)}, 倒序: {(i2, p2)}, 最终临界位置: {l}, 周期: {p}")
    return l, p


def small_period(pattern: str):
    n = len(pattern)
    l, p = cf(pattern)

    if l < n // 2 and l > 0 and l <= p:
        segment = pattern[l:l+p]
        if segment.endswith(pattern[0:l]):
            return p

    return max(l, n - l) + 1


def positions_bis(text: str, pattern: str) -> list[int]:
    n = len(pattern)
    m = len(text)

    # 获取临界位置和周期
    l, p = cf(pattern)
    q = small_period(pattern)   # q := max(l, n – l) + 1

    print(f"临界位置 l = {l}, 周期 p = {p}, 移动步长 q = {q}")

    results: list[int] = []
    pos = 0

    while pos + n <= m:
        i = l + 1
        while i <= n and pattern[i-1] == text[pos + i - 1]:  # 匹配xr的部分
            i += 1

        if i <= n:
            shift = i - l
            pos += shift
        else:   # xr部分匹配，扫描xl
            j = l
            while j > 0 and pattern[j-1] == text[pos + j - 1]:
                j -= 1
            if j == 0:
                results.append(pos)

            # 无论是否匹配，都移动q的距离
            pos += q

    return results


if __name__ == "__main__":
    text = "caabbaabbacabc"
    pattern = "abbaa"
    print(f"文本: {text}")
    print(f"模式: {pattern}")

    all_positions = positions_bis(text, pattern)
    print(f"所有位置: {all_positions}")
