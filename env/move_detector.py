# guandan/env/move_detector.py
from collections import Counter
from .settings import *
from .utils import get_logical_rank, is_wild_card

class MoveDetector:
    @staticmethod
    def get_move_type(card_ids, cur_level_rank_idx):
        """
        输入牌ID列表，返回 {'type': TYPE_xxx, 'rank': 主牌逻辑点数, 'count': 张数}
        如果是不合法的牌型，返回 None
        """
        if not card_ids:
            return {'type': TYPE_PASS, 'rank': 0, 'count': 0}
            
        n = len(card_ids)
        
        # 1. 预处理
        wild_cards = []
        normal_cards = []
        normal_ranks = []   # 逻辑点数 (级牌=15)
        natural_ranks = []  # 自然点数 (级牌=原值，用于顺子检测)
        
        for cid in card_ids:
            if is_wild_card(cid, cur_level_rank_idx):
                wild_cards.append(cid)
            else:
                normal_cards.append(cid)
                
                # A. 计算逻辑点数 (用于比大小、判断炸弹/三带二)
                logic_r = get_logical_rank(cid, cur_level_rank_idx)
                normal_ranks.append(logic_r)
                
                # B. 计算自然点数 (用于顺子/连对)
                # 大小王保持 20/30，普通牌/级牌还原为 2-14
                # cid % 13: 0->2, 12->A
                if cid >= 104: 
                    natural_ranks.append(logic_r)
                else:
                    natural_ranks.append(cid % 13 + 2)
                
        num_wild = len(wild_cards)
        normal_ranks.sort()
        natural_ranks.sort()
        
        # --- 特殊检测：四大天王 ---
        if n == 4 and all(cid >= 104 for cid in card_ids):
             return {'type': TYPE_KING_BOMB, 'rank': 30, 'count': 4}

        # --- 炸弹检测 (4张及以上) ---
        if n >= 4:
            if len(set(normal_ranks)) <= 1:
                if not normal_ranks: rank = 15 
                else: rank = normal_ranks[0]
                
                # 同花顺检测 (优先判定)
                is_sf = MoveDetector._check_straight_flush(card_ids, cur_level_rank_idx)
                if is_sf:
                    return {'type': TYPE_STRAIGHT_FLUSH, 'rank': is_sf, 'count': 5}
                
                return {'type': TYPE_BOMB, 'rank': rank, 'count': n}

        # --- 单张 ---
        if n == 1:
            rank = normal_ranks[0] if normal_ranks else 15
            return {'type': TYPE_SINGLE, 'rank': rank, 'count': 1}
        
        # --- 对子 ---
        if n == 2:
            if len(set(normal_ranks)) <= 1:
                rank = normal_ranks[0] if normal_ranks else 15
                return {'type': TYPE_PAIR, 'rank': rank, 'count': 2}

        # --- 三张 ---
        if n == 3:
            if len(set(normal_ranks)) <= 1:
                rank = normal_ranks[0] if normal_ranks else 15
                return {'type': TYPE_TRIPLE, 'rank': rank, 'count': 3}

        # --- 三带二 (5张) ---
        if n == 5:
            # 优先判定三带二
            res = MoveDetector._check_triple_pair(normal_ranks, num_wild)
            if res:
                return {'type': TYPE_TRIPLE_PAIR, 'rank': res, 'count': 5}
            
            # 顺子检测 (尝试逻辑点数 OR 自然点数)
            # 1. 尝试逻辑顺子 (例如 10-J-Q-K-A)
            res_str = MoveDetector._check_consecutive(normal_ranks, num_wild, length=5)
            # 2. 如果失败，尝试自然顺子 (例如 2-3-4-5-6，此时2作为级牌是自然点数)
            if not res_str:
                res_str = MoveDetector._check_consecutive(natural_ranks, num_wild, length=5)
            
            if res_str:
                # 再次确认是否同花顺
                is_sf = MoveDetector._check_straight_flush(card_ids, cur_level_rank_idx)
                if is_sf:
                    return {'type': TYPE_STRAIGHT_FLUSH, 'rank': is_sf, 'count': 5}
                return {'type': TYPE_STRAIGHT, 'rank': res_str, 'count': 5}

        # --- 三连对 (6张) ---
        if n == 6:
            # 尝试逻辑点数 (如 JJQQKK)
            res = MoveDetector._check_tube(normal_ranks, num_wild)
            # 尝试自然点数 (如 223344，2是级牌)
            if not res:
                res = MoveDetector._check_tube(natural_ranks, num_wild)
                
            if res:
                return {'type': TYPE_TUBE, 'rank': res, 'count': 6}
            
            # 钢板 (两个三张)
            res_plate = MoveDetector._check_plate(normal_ranks, num_wild)
            if not res_plate:
                res_plate = MoveDetector._check_plate(natural_ranks, num_wild)
                
            if res_plate:
                return {'type': TYPE_PLATE, 'rank': res_plate, 'count': 6}

        return None

    @staticmethod
    def _check_triple_pair(ranks, num_wild):
        """检测三带二"""
        counts = Counter(ranks)
        possible_triples = set(ranks)
        if not possible_triples: possible_triples = {15}

        for t_rank in possible_triples:
            needed_for_triple = max(0, 3 - counts[t_rank])
            remain_wild = num_wild - needed_for_triple
            if remain_wild < 0: continue
            
            remain_ranks = list(ranks)
            for _ in range(counts[t_rank]): remain_ranks.remove(t_rank)
            
            if len(remain_ranks) == 0:
                 if remain_wild == 2: return t_rank
            else:
                 if len(set(remain_ranks)) == 1:
                     p_rank = remain_ranks[0]
                     if p_rank != t_rank:
                         needed_for_pair = max(0, 2 - counts[p_rank])
                         if remain_wild >= needed_for_pair:
                             return t_rank
        return None

    @staticmethod
    def _check_consecutive(ranks, num_wild, length=5):
        """检测连续顺子"""
        if not ranks: 
            # 全是百搭牌，直接当作最大的 A 顺子 (rank 14)
            return 14 
        
        unique_ranks = sorted(list(set(ranks)))
        if len(unique_ranks) != len(ranks): return None # 有对子
        
        # 顺子不能含王 (除非王被当做百搭处理了，但这里的 ranks 是非百搭牌)
        if any(r >= 20 for r in unique_ranks): return None
        
        min_r = unique_ranks[0]
        max_r = unique_ranks[-1]
        
        # --- 情况A: 普通顺子 (2-3-4-5-6) ---
        span = max_r - min_r
        needed = (span + 1) - len(unique_ranks)
        if span < length and num_wild >= needed:
            # 计算顺子最大值
            top_rank = max_r + (num_wild - needed)
            if top_rank > 14: top_rank = 14
            return top_rank
            
        # --- 情况B: A-2-3-4-5 (A当1用) ---
        # 此时 A 是 14，出现在列表最后
        if 14 in unique_ranks:
            # 将 14 临时改为 1，重新排序检查
            temp_ranks = sorted([1 if r == 14 else r for r in unique_ranks])
            min_r2 = temp_ranks[0]
            max_r2 = temp_ranks[-1]
            span2 = max_r2 - min_r2
            needed2 = (span2 + 1) - len(temp_ranks)
            
            if span2 < length and num_wild >= needed2:
                # A-2-3-4-5 的最大牌是 5
                top_rank = max_r2 + (num_wild - needed2)
                return top_rank

        return None

    @staticmethod
    def _check_tube(ranks, num_wild):
        """检测三连对"""
        counts = Counter(ranks)
        unique_ranks = sorted(counts.keys())
        
        if len(unique_ranks) > 3: return None
        if any(r >= 20 for r in unique_ranks): return None
        
        # 遍历起始点 (支持自然点数的连对，也支持 A23 连对)
        # 这里简化只检测普通连对
        for start in range(2, 13): # 2 到 Q (连对最大 AA KK QQ -> 14)
            needed = 0
            for r in range(start, start+3):
                c = counts.get(r, 0)
                needed += max(0, 2 - c)
            
            if num_wild >= needed:
                return start + 2
        
        # 特殊: A-A-2-2-3-3 (A当1) -> 最大是3
        # 如果 ranks 包含 14, 2, 3，可考虑特殊处理。暂时略过以保持简洁。
        return None

    @staticmethod
    def _check_plate(ranks, num_wild):
        """检测钢板 (333444)"""
        counts = Counter(ranks)
        unique_ranks = sorted(counts.keys())
        if len(unique_ranks) > 2: return None
        
        for start in range(2, 14): 
            needed = 0
            for r in range(start, start+2):
                c = counts.get(r, 0)
                needed += max(0, 3 - c)
            if num_wild >= needed:
                return start + 1
        return None

    @staticmethod
    def _check_straight_flush(card_ids, cur_level_rank_idx):
        """同花顺检测"""
        # 1. 检查花色
        suits = []
        for cid in card_ids:
            if not is_wild_card(cid, cur_level_rank_idx):
                # 大小王不能参与同花顺
                if cid >= 104: return None
                suits.append((cid % 52) // 13)
        
        if len(set(suits)) > 1: return None
        
        # 2. 检查点数 (尝试逻辑点数 AND 自然点数)
        # 提取非百搭牌的 ID
        normal_ids = [cid for cid in card_ids if not is_wild_card(cid, cur_level_rank_idx)]
        wild_count = len(card_ids) - len(normal_ids)
        
        # 方案 A: 逻辑点数 (处理 10-J-Q-K-A)
        logic_ranks = [get_logical_rank(cid, cur_level_rank_idx) for cid in normal_ids]
        res = MoveDetector._check_consecutive(logic_ranks, wild_count, length=5)
        if res: return res
        
        # 方案 B: 自然点数 (处理 2-3-4-5-6，其中2是级牌)
        natural_ranks = [cid % 13 + 2 for cid in normal_ids] # 已经排除了王
        res = MoveDetector._check_consecutive(natural_ranks, wild_count, length=5)
        
        return res