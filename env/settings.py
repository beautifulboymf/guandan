# guandan/env/settings.py

SUITS = ['S', 'H', 'C', 'D']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# 基础物理点数 (2-A)
RANK_VALUE = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    'SB': 20, 'HR': 30  # 小王，大王 (BJ/RJ)
}

# 牌型定义
TYPE_PASS = 0           # 过
TYPE_SINGLE = 1         # 单张
TYPE_PAIR = 2           # 对子
TYPE_TRIPLE = 3         # 三张 (不带) - *注：掼蛋通常不许三不带，除非最后一手，这里暂留*
TYPE_TRIPLE_PAIR = 4    # 三带二 (例如 33344)
TYPE_STRAIGHT = 5       # 顺子 (5张)
TYPE_TUBE = 6           # 三连对 (例如 334455)
TYPE_PLATE = 7          # 钢板 (两个连续三张，例如 333444)
TYPE_STRAIGHT_FLUSH = 8 # 同花顺 (最大炸弹之一)
TYPE_BOMB = 9           # 炸弹 (4张及以上)
TYPE_KING_BOMB = 10     # 四大天王 (4个王)