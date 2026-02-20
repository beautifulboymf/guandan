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
        
        # 解析桌面上的最后一手牌
        last_move_info = None
        if infoset.last_move and infoset.last_pid != self.player_id:
            last_move_info = MoveDetector.get_move_type(infoset.last_move, cur_level_idx)
            
        # 获取所有合法动作
        generator = MoveGenerator(infoset.my_hand, cur_level_idx)
        legal_actions = generator.get_legal_actions(last_move_info)
        
        # 随机挑一个
        return random.choice(legal_actions)

class HeuristicAgent:
    """
    启发式规则智能体 (基准 Bot)
    策略：只要有牌能管上上家，就绝对不 PASS。
    跟牌时：挑刚好能管上的、代价最小的那一组牌出。
    先手时：随机挑选一种手里的牌型组合（单张/对子/顺子等），打出该组合中最小的，更具人性化。
    """
    def __init__(self, player_id):
        self.player_id = player_id

    def act(self, infoset):
        cur_level_idx = infoset.cur_level - 2
        
        # 1. 解析上家出牌
        last_move_info = None
        if infoset.last_move and infoset.last_pid != self.player_id:
            last_move_info = MoveDetector.get_move_type(infoset.last_move, cur_level_idx)
            
        # 2. 生成合法动作
        generator = MoveGenerator(infoset.my_hand, cur_level_idx)
        legal_actions = generator.get_legal_actions(last_move_info)

        # 剥离出所有的实体动作 (去掉 PASS)
        real_actions = [a for a in legal_actions if a != []]

        # 如果真的没有任何牌能打，只能 PASS
        if not real_actions:
            return []

        # 3. 对合法实体动作进行精确打分排序
        scored_actions = []
        for action in real_actions:
            move_info = MoveDetector.get_move_type(action, cur_level_idx)
            t = move_info['type']
            r = move_info['rank']
            c = move_info['count']
            
            # 精准量化牌的“代价”: (是否是炸弹, 张数/威力大小, 逻辑点数)
            if t == TYPE_KING_BOMB:
                score = (1, 10, 30)      
            elif t == TYPE_STRAIGHT_FLUSH:
                score = (1, 9, r)        
            elif t == TYPE_BOMB:
                score = (1, c, r)        
            else:
                score = (0, c, r)        
                
            # 我们把 move_info 也存进去，方便后续提炼牌型
            scored_actions.append((score, action, move_info))
            
        # 4. 决策逻辑
        if last_move_info is None:
            # ==========================================
            # 【人性化先手逻辑】：随机出组合牌
            # ==========================================
            # 把炸弹和普通牌分开
            non_bombs = [item for item in scored_actions if item[0][0] == 0]
            
            if non_bombs:
                # 看看当前手里能凑出哪些普通牌型 (如 TYPE_SINGLE, TYPE_PAIR, TYPE_STRAIGHT 等)
                available_types = list(set(item[2]['type'] for item in non_bombs))
                
                # 随机挑一种牌型来探路！(比如随机决定这把先出顺子)
                chosen_type = random.choice(available_types)
                
                # 筛选出选定牌型的所有动作
                candidates = [item for item in non_bombs if item[2]['type'] == chosen_type]
                
                # 在这种牌型里，挑点数最小的打出去 (按 rank 排序)
                candidates.sort(key=lambda x: x[0][2])
                return candidates[0][1]
            else:
                # 如果手里全是炸弹，那就出最小的炸弹
                scored_actions.sort(key=lambda x: x[0])
                return scored_actions[0][1]
                
        else:
            # ==========================================
            # 【跟牌逻辑】：保持原样，最小代价压制
            # ==========================================
            scored_actions.sort(key=lambda x: x[0])
            return scored_actions[0][1]