"""
利用DP来计算编辑距离
"""

def ed(s1: str, s2: str):
    m = len(s1)
    n = len(s2)

    # DP 表与初始化
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            a = s1[i - 1]
            b = s2[j - 1]
            if a == b:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    return dp


if __name__ == "__main__":
    text = "abcd"
    pattern = "cbc"

    table = ed(text, pattern)
    print(table)