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
        
        # 1. 预处理：分离百搭牌和普通牌，计算逻辑点数
        wild_cards = []
        normal_cards = []
        normal_ranks = []
        
        for cid in card_ids:
            if is_wild_card(cid, cur_level_rank_idx):
                wild_cards.append(cid)
            else:
                normal_cards.append(cid)
                normal_ranks.append(get_logical_rank(cid, cur_level_rank_idx))
                
        num_wild = len(wild_cards)
        normal_ranks.sort()
        
        # --- 特殊检测：四大天王 ---
        if n == 4 and all(cid >= 104 for cid in card_ids):
             return {'type': TYPE_KING_BOMB, 'rank': 30, 'count': 4}

        # --- 炸弹检测 (4张及以上) ---
        # 逻辑：如果只有一种逻辑点数（加上百搭牌后），就是炸弹
        if n >= 4:
            if len(set(normal_ranks)) <= 1:
                # 确定炸弹的主点数
                if not normal_ranks: # 全是百搭牌 (罕见，当作级牌炸弹)
                    rank = 15 
                else:
                    rank = normal_ranks[0]
                
                # 同花顺检测 (5张)
                # 如果是5张，且是炸弹，需要检查是否同花顺
                # 注意：掼蛋中同花顺必须是同花色且连续。
                # 这里的逻辑稍微复杂，先简单判定为普通炸弹，后续加同花顺判定
                is_sf = MoveDetector._check_straight_flush(card_ids, cur_level_rank_idx)
                if is_sf:
                    return {'type': TYPE_STRAIGHT_FLUSH, 'rank': is_sf, 'count': 5}
                
                return {'type': TYPE_BOMB, 'rank': rank, 'count': n}

        # --- 单张 ---
        if n == 1:
            return {'type': TYPE_SINGLE, 'rank': get_logical_rank(card_ids[0], cur_level_rank_idx), 'count': 1}

        # --- 对子 ---
        if n == 2:
            if len(set(normal_ranks)) <= 1: # 百搭牌可以配合任意牌
                rank = normal_ranks[0] if normal_ranks else 15
                return {'type': TYPE_PAIR, 'rank': rank, 'count': 2}

        # --- 三张 (不带) ---
        if n == 3:
            if len(set(normal_ranks)) <= 1:
                rank = normal_ranks[0] if normal_ranks else 15
                return {'type': TYPE_TRIPLE, 'rank': rank, 'count': 3}

        # --- 三带二 (5张) ---
        if n == 5:
            res = MoveDetector._check_triple_pair(normal_ranks, num_wild)
            if res:
                return {'type': TYPE_TRIPLE_PAIR, 'rank': res, 'count': 5}
            
            # 顺子检测 (5张)
            res_str = MoveDetector._check_consecutive(normal_ranks, num_wild, length=5)
            if res_str:
                # 再次检查是否同花顺
                is_sf = MoveDetector._check_straight_flush(card_ids, cur_level_rank_idx)
                if is_sf:
                    return {'type': TYPE_STRAIGHT_FLUSH, 'rank': is_sf, 'count': 5}
                return {'type': TYPE_STRAIGHT, 'rank': res_str, 'count': 5}

        # --- 三连对 (6张) ---
        if n == 6:
            res = MoveDetector._check_tube(normal_ranks, num_wild)
            if res:
                return {'type': TYPE_TUBE, 'rank': res, 'count': 6}
            
            # 钢板 (两个三张)
            res_plate = MoveDetector._check_plate(normal_ranks, num_wild)
            if res_plate:
                return {'type': TYPE_PLATE, 'rank': res_plate, 'count': 6}

        return None # 不合规

    @staticmethod
    def _check_triple_pair(ranks, num_wild):
        """检测三带二，返回三张部分的rank"""
        # 统计频次
        counts = Counter(ranks)
        # 目标：形成一个3张和一个2张
        # 策略：遍历所有可能的“三张”的主牌
        possible_triples = set(ranks)
        if not possible_triples: possible_triples = {15} # 全是百搭

        for t_rank in possible_triples:
            needed_for_triple = max(0, 3 - counts[t_rank])
            
            # 剩余的百搭牌
            remain_wild = num_wild - needed_for_triple
            if remain_wild < 0: continue
            
            # 检查剩下的是否能凑成对子
            # 剩下的牌排除掉组成三张的那部分
            remain_ranks = list(ranks)
            for _ in range(counts[t_rank]): remain_ranks.remove(t_rank) # 移除已用的
            # 实际上只要移除 min(3, count) 个，但为了简单，直接看剩余逻辑
            # 这里逻辑简化：三带二必须是 3张A + 2张B
            
            # 如果剩余牌是空的（说明之前只有一种牌，比如 AAAA+Wild），那就是炸弹逻辑，不走这里
            # 但如果是 AAA + BB，剩余 BB
            if len(remain_ranks) == 0:
                 if remain_wild == 2: return t_rank # AAA + WW
            else:
                 # 剩下的牌必须只有一种点数
                 if len(set(remain_ranks)) == 1:
                     p_rank = remain_ranks[0]
                     if p_rank != t_rank: # 必须不同点数
                         needed_for_pair = max(0, 2 - counts[p_rank])
                         if remain_wild >= needed_for_pair:
                             return t_rank
        return None

    @staticmethod
    def _check_consecutive(ranks, num_wild, length=5):
        """检测连续顺子，返回最大牌的rank"""
        if not ranks: return 14 # 假设全是百搭，当作A顶天
        
        # 去重并排序，顺子不能有重复点数 (除非用来填空，但 ranks 已经是排好序的非百搭牌)
        unique_ranks = sorted(list(set(ranks)))
        if len(unique_ranks) != len(ranks): return None # 有对子肯定不是顺子
        
        # 顺子中不能包含大小王 (逻辑分 20, 30)
        if any(r > 14 and r != 15 for r in unique_ranks): return None 
        # 掼蛋规则：A可以做 1 2 3 4 5 的尾 (14), 也可以做 A 2 3 4 5 (这时候A当1用)
        # 这里简化处理：通常 A 2 3 4 5 是最小顺子， 10 J Q K A 是最大
        
        min_r = unique_ranks[0]
        max_r = unique_ranks[-1]
        
        # 跨度必须小于长度
        span = max_r - min_r
        if span >= length: return None
        
        # 需要填补的空缺
        needed = (span + 1) - len(unique_ranks)
        
        if num_wild >= needed:
            # 牌够凑
            # 计算最大点数。
            # 如果当前是 2 3，有3个百搭，最大是 6 (2 3 4 5 6)
            # 也就是 max_r + (num_wild - needed)
            top_rank = max_r + (num_wild - needed)
            if top_rank > 14: top_rank = 14 # A封顶
            return top_rank
        return None

    @staticmethod
    def _check_tube(ranks, num_wild):
        """检测三连对 (334455)"""
        # 简化版：暂不支持复杂的百搭补牌，假设百搭只能补缺
        # 真正的逻辑需要回溯，这里写一个简易判断
        # 统计每个点数的数量
        counts = Counter(ranks)
        unique_ranks = sorted(counts.keys())
        
        if len(unique_ranks) > 3: return None
        if any(r > 14 and r!=15 for r in unique_ranks): return None
        
        # 遍历可能的起始点
        for start in range(2, 13): # 2 到 Q
            needed = 0
            for r in range(start, start+3):
                # 这里的r需要考虑级牌是15的情况吗？三连对通常不带级牌玩，除非级牌当普通牌
                # 掼蛋规定：级牌如果是本身点数，参与顺子；如果当百搭，不参与。
                # 这里的 ranks 是逻辑点数，级牌已经是 15 了。
                # 如果顺子/三连对里有 15，说明级牌没当百搭用，而是当“级牌本级”用？
                # 实际上掼蛋级牌在顺子里通常按物理位置排。
                # *重要简化*：为了不把逻辑搞太复杂，这里假设三连对必须是连续的逻辑值
                # 且不包含级牌(15)，除非级牌正好卡在中间（难处理）
                # 暂时只支持非级牌的连对
                c = counts.get(r, 0)
                needed += max(0, 2 - c)
            
            if num_wild >= needed:
                return start + 2 # 返回最大那对的rank
        return None

    @staticmethod
    def _check_plate(ranks, num_wild):
        """检测钢板 (333444)"""
        counts = Counter(ranks)
        unique_ranks = sorted(counts.keys())
        if len(unique_ranks) > 2: return None
        
        for start in range(2, 14): # 2 到 K
            needed = 0
            for r in range(start, start+2):
                c = counts.get(r, 0)
                needed += max(0, 3 - c)
            if num_wild >= needed:
                return start + 1
        return None

    @staticmethod
    def _check_straight_flush(card_ids, cur_level_rank_idx):
        """同花顺检测 (简易版)"""
        # 必须全是同一花色 (除去百搭)
        suits = []
        for cid in card_ids:
            if not is_wild_card(cid, cur_level_rank_idx):
                suits.append((cid % 52) // 13)
        
        if len(set(suits)) > 1: return None # 花色杂乱
        
        # 剩下的逻辑和顺子一样
        ranks = [get_logical_rank(cid, cur_level_rank_idx) for cid in card_ids if not is_wild_card(cid, cur_level_rank_idx)]
        num_wild = len(card_ids) - len(ranks)
        
        return MoveDetector._check_consecutive(ranks, num_wild, length=5)