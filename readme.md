## 概述
bilibili上算法视频的相关代码
大部分代码都是Chatgpt或者Deepseek写的，自己在此基础之上做了一些校验、注释和bug修复

python版本为3.12.11
numpy版本为1.26.4

## 算法
### 模式匹配 Pattern Matching
- [KMP(Knuth-Morris-Pratt)算法](pattern_matching/kmp.py)
- [Horspool(boyer-moore-horspool)算法](pattern_matching/bmh.py)
- [Boyer-Moore算法](pattern_matching/bm.py)
- [Rabin-Karp算法](pattern_matching/rk.py)
- [Bitap算法(shift-or/shift-and)](pattern_matching/bitap.py)
- [Aho-Corasick算法](pattern_matching/ac.py)
- [Suffix Automation](pattern_matching/sa.py)
- [Two-way string matching算法](pattern_matching/twsm.py)

### 字符串近似匹配 Approximate String Matching
- [Edit Distance](fuzzy_string_search/ed.py)

### 寻路算法 Path Finding
- [BFS搜索](pathfinding/bfs.py)
- [Dijkstra算法](pathfinding/dijkstra.py)
- [Conflict-Based Search](pathfinding/cbs.py)
- [Meta-Agent CBS](pathfinding/macbs.py)

## 数据结构
### 堆 Heap
- [二叉堆(Binary Heap)](heap/binary.py)
- [斐波那契堆(Fibonacci Heap)](heap/fib.py)
- [配对堆(Pairing Heap)](heap/pairing.py)