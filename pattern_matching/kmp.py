"""
Knuth-Morris-Pratt 算法
代码参考了: https://www.geeksforgeeks.org/dsa/kmp-algorithm-for-pattern-searching/
修改了一些变量名，并加入了一些注释
"""


def get_table(pattern: str) -> list[int]:
    """
    计算LPS(Longest Prefix Suffix)表
    即, 首尾相同的部分的长度 [xxx]...[xxx]
    """
    size: int = 0
    len_pattern = len(pattern)
    lps: list[int] = [0 for _ in range(len_pattern)]    # 初始化

    i = 1
    while i < len_pattern:
        # 如果检测到首尾相同的部分, 增加 size
        # 并将其记录到对应的LPS位置
        if pattern[i] == pattern[size]:
            size += 1
            lps[i] = size
            i += 1
        else:
            if size != 0:
                size = lps[size - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(pattern: str, text: str):
    len_text = len(text)
    len_pattern = len(pattern)

    results: list[int] = []  # 匹配到的位置(索引)
    lps = get_table(pattern)
    print(f"{pattern}\n{lps}")

    i = 0   # 文本所在的位置
    p = 0   # pattern 所在的位置
    while i < len_text:
        # 过程可视化
        print(f"当前i:{i}, p:{p}")
        print(text[:i]+"("+text[i]+")"+text[i+1:])
        print(pattern[:p]+"("+pattern[p]+")"+pattern[p+1:])

        # 如果有字符相等, 匹配下一个字符
        if text[i] == pattern[p]:
            i += 1
            p += 1

            if p == len_pattern:  # 全部相等
                results.append(i - p)
                p = lps[p - 1]

        # 如果不相等, 利用LPS表的信息, 去掉重复匹配可能的位置
        else:
            # 如果指针 p 不在开头的位置，那么, 移动 p 的指针后再进行一次比较
            if p != 0:
                p = lps[p - 1]
                print(f"p --> {p}")
            else:
                i += 1
    return results


if __name__ == "__main__":
    text = "abcabddabcaabcc"
    pattern = "abcaabc"
    print(f"文本: {text}")
    print(f"模式: {pattern}")

    res = kmp_search(pattern, text)
    print(f"所有位置: {res}")
