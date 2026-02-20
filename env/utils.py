from .settings import SUITS, RANKS

def get_card_name(card_id):
    """将数字ID转换为人类可读字符串，例如 0 -> S2 (黑桃2)"""
    if card_id < 104:
        # 普通牌
        suit_idx = (card_id % 52) // 13
        rank_idx = card_id % 13
        return f"{SUITS[suit_idx]}{RANKS[rank_idx]}"
    elif card_id in [104, 105]:
        return "SB" # Small Joker (小王)
    elif card_id in [106, 107]:
        return "HR" # Red Joker (大王)
    return "Unknown"

def get_card_sort_key(card_id):
    """
    辅助函数：定义排序规则
    1. 大王 > 小王 > 普通牌
    2. 普通牌按点数 (2-A) 排序
    3. 同点数按花色排序
    """
    if card_id in [106, 107]: # 大王
        return (100, card_id)
    if card_id in [104, 105]: # 小王
        return (99, card_id)
    
    # 普通牌：先看点数 (rank)，再看花色 (suit)
    # 0-12 是 2-A
    rank_idx = card_id % 13
    suit_idx = (card_id % 52) // 13
    return (rank_idx, suit_idx)

def format_hand(cards_ids):
    """将手牌ID列表排序并转换为可视化字符串"""
    if not cards_ids:
        return "[]"
        
    # 复制并排序
    sorted_ids = sorted(cards_ids, key=get_card_sort_key)
    
    display = []
    for cid in sorted_ids:
        display.append(get_card_name(cid))
    return "[" + " ".join(display) + "]"

def parse_input_string(input_str, hand_ids):
    """
    解析用户输入的字符串
    支持: 'H2, S3' 或 'H2 S3' (空格或逗号分隔)
    """
    normalized_str = input_str.replace(',', ' ')
    target_names = [x.strip().upper() for x in normalized_str.split() if x.strip()]
    
    played_ids = []
    temp_hand = hand_ids.copy()
    
    for name in target_names:
        name_upper = name.upper()
        
        # 别名处理：移除 SJ，避免与 黑桃J (Spade Jack) 冲突！
        # 增加拼音别名 XW(小王), DW(大王)
        if name_upper in ['BJ', 'HR', 'DW']: 
            name_upper = 'HR'
        elif name_upper in ['SB', 'XW']: 
            name_upper = 'SB'
        
        found = False
        for cid in temp_hand:
            # 统一转成大写比对
            if get_card_name(cid).upper() == name_upper:
                played_ids.append(cid)
                temp_hand.remove(cid)
                found = True
                break
                
        if not found:
            raise ValueError(f"手牌中没有: {name_upper}")
            
    return played_ids

def get_logical_rank(card_id, cur_level_rank_idx):
    """
    获取牌在当前级数下的逻辑大小。
    cur_level_rank_idx: 当前打几 (0代表2, 1代表3... 12代表A)
    
    逻辑大小定义：
    2-A (非级牌): 2-14
    级牌: 15
    小王: 20
    大王: 30
    """
    if card_id >= 106: return 30 # 大王
    if card_id >= 104: return 20 # 小王
    
    # 物理点数 (0-12 代表 2-A)
    rank_idx = card_id % 13
    
    # 如果是当前级牌
    if rank_idx == cur_level_rank_idx:
        return 15
        
    # 普通牌映射到 2-14
    # 注意：如果当前打5，那么4的逻辑分是4，6的逻辑分是6。
    # 这里的 rank_idx+2 对应 RANK_VALUE
    return rank_idx + 2

def is_wild_card(card_id, cur_level_rank_idx):
    """判断是否为逢人配（当前级数的红心牌）"""
    if card_id >= 104: return False
    rank_idx = card_id % 13
    suit_idx = (card_id % 52) // 13
    # 假设 settings.py 中 SUITS = ['S', 'H', 'C', 'D']，下标1是红心
    return rank_idx == cur_level_rank_idx and suit_idx == 1