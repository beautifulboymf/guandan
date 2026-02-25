import os
import sys
import random

# 确保导入路径正确
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.settings import *
from env.move_detector import MoveDetector
from env.move_generator import MoveGenerator
from env.utils import format_hand

class RandomAgent:
    """纯随机出牌的傻瓜智能体 (用于兜底测试)"""
    def __init__(self, player_id):
        self.player_id = player_id

    def act(self, infoset):
        cur_level_idx = infoset.cur_level - 2
        last_move_info = None
        if infoset.last_move and infoset.last_pid != self.player_id:
            last_move_info = MoveDetector.get_move_type(infoset.last_move, cur_level_idx)
        generator = MoveGenerator(infoset.my_hand, cur_level_idx)
        legal_actions = generator.get_legal_actions(last_move_info)
        return random.choice(legal_actions)


class HeuristicAgent:
    """
    【高阶进化版】启发式规则智能体 (大师 Bot)
    包含特性：
    1. 队友意识：队友牌少时主动喂牌（单/对）；队友出大牌时主动让路。
    2. 牌型梳理：寻找有“大、小”重复的组合（如两组钢板），打出小的骗炸弹/回手。
    """
    def __init__(self, player_id):
        self.player_id = player_id

    def act(self, infoset):
        cur_level_idx = infoset.cur_level - 2
        me = self.player_id
        
        # 获取队友的 ID 和剩余牌数
        teammate_id = (me + 2) % 4
        teammate_cards_num = infoset.others_num.get(teammate_id, 27)
        
        # 1. 解析上家出牌
        last_move_info = None
        if infoset.last_move and infoset.last_pid != me:
            last_move_info = MoveDetector.get_move_type(infoset.last_move, cur_level_idx)
            
        # 2. 生成合法动作
        generator = MoveGenerator(infoset.my_hand, cur_level_idx)
        legal_actions = generator.get_legal_actions(last_move_info)

        real_actions = [a for a in legal_actions if a != []]

        # 如果真的没有任何牌能打，只能 PASS
        if not real_actions:
            return []

        # 3. 对合法实体动作进行精确打分排序
        scored_actions = []
        for action in real_actions:
            move_info = MoveDetector.get_move_type(action, cur_level_idx)
            if not move_info: continue
            t = move_info['type']
            r = move_info['rank']
            c = move_info['count']
            
            # 代价打分机制 (是否是炸弹, 张数/威力大小, 逻辑点数)
            if t == TYPE_KING_BOMB:
                score = (1, 10, 30)      
            elif t == TYPE_STRAIGHT_FLUSH:
                score = (1, 9, r)        
            elif t == TYPE_BOMB:
                score = (1, c, r)        
            else:
                score = (0, c, r)        
                
            scored_actions.append((score, action, move_info))
            
        # ==========================================
        # 4. 核心决策大脑
        # ==========================================
        if last_move_info is None:
            # 【场景 A：先手出牌】
            non_bombs = [item for item in scored_actions if item[0][0] == 0]
            
            if non_bombs:
                # --- 机制 1：残局喂牌 ---
                # 如果队友只剩 <= 5 张牌，极大可能只需要单张或对子就能走掉
                if teammate_cards_num <= 5:
                    feed_candidates = [item for item in non_bombs if item[2]['type'] in [TYPE_SINGLE, TYPE_PAIR]]
                    if feed_candidates:
                        # 挑最小的单张或对子打，送队友走
                        feed_candidates.sort(key=lambda x: x[0][2])
                        return feed_candidates[0][1]
                
                # --- 机制 2：高级牌型梳理 (诱导/回手) ---
                # 将可出的牌按类型分组 (比如收集手里所有的顺子、钢板、三带二)
                type_lists = {}
                for item in non_bombs:
                    t = item[2]['type']
                    if t not in type_lists: type_lists[t] = []
                    type_lists[t].append(item)
                    
                # 寻找手里拥有 2 组及以上的牌型 (就像你说的：有大有小，可以回手！)
                multi_types = [t for t, acts in type_lists.items() if len(acts) >= 2]
                
                if multi_types:
                    # 在这些能回手的牌型里，挑出最“庞大/复杂”的 (比如有钢板有对子，优先发钢板)
                    best_type = max(multi_types, key=lambda t: type_lists[t][0][2]['count'])
                else:
                    # 如果没有重复的，就顺着手牌，把张数最多的组合打掉 (比如唯一的长顺子)
                    best_type = max(type_lists.keys(), key=lambda t: type_lists[t][0][2]['count'])
                    
                candidates = type_lists[best_type]
                
                # 核心：出该组里【最小】的那一个！探路、消耗对手炸弹、或等待用手里大的回手！
                candidates.sort(key=lambda x: x[0][2])
                return candidates[0][1]
            else:
                # 绝境：手里全剩炸弹了，出最小的炸弹
                scored_actions.sort(key=lambda x: x[0])
                return scored_actions[0][1]
                
        else:
            # 【场景 B：跟牌压制】
            if infoset.last_pid == teammate_id:
                # --- 机制 3：极度聪明的队友让路 ---
                # 如果队友快跑完了，直接过，死也不挡路！
                if teammate_cards_num <= 5:
                    return []
                # 如果队友出的牌已经具备统治力（逻辑点数 >= 11，即 J,Q,K,A），保留实力，让他飞！
                if last_move_info['rank'] >= 11:
                    return []
                    
            # 正常压制对手：在能管上的牌里，挑代价最小的（最抠门）
            scored_actions.sort(key=lambda x: x[0])
            return scored_actions[0][1]