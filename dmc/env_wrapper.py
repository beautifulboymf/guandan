import numpy as np
import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.game import GameEnv
from env.move_generator import MoveGenerator
from env.move_detector import MoveDetector

MAX_HISTORY = 15 # 提取最近 15 手历史记录，专供 Attention 模块使用

def cards2array(card_ids):
    """
    将物理卡牌 ID 转换为 108 维的 0/1 矩阵
    前54维代表该点数是否有第一张，后54维代表是否有第二张
    """
    vec = np.zeros(108, dtype=np.float32)
    if not card_ids: return vec
        
    from collections import Counter
    type_ids = []
    for cid in card_ids:
        if cid < 104: type_ids.append(cid % 52)
        elif cid in [104, 105]: type_ids.append(52) # 小王归入52
        else: type_ids.append(53)                   # 大王归入53
            
    for tid, count in Counter(type_ids).items():
        if count >= 1: vec[tid] = 1.0
        if count == 2: vec[tid + 54] = 1.0
    return vec

class GuandanEnvWrapper:
    """面向 DMC 与 Transformer/Cross-Attention 的环境包装器"""
    def __init__(self):
        self.env = GameEnv()

    def reset(self, current_level=2):
        infoset = self.env.reset(current_level)
        return self._get_obs(infoset)

    def step(self, action):
        next_infoset, reward, done, result_info = self.env.step(action)
        obs = self._get_obs(next_infoset) if not done else None
        return obs, reward, done, result_info

    def _get_obs(self, infoset):
        me = infoset.player_id
        cur_level_idx = infoset.cur_level - 2
        
        # ==========================================
        # 1. 计算 Context Feature (算牌底座 + 场面状态) - 205 维
        # ==========================================
        played_cards = []
        for item in infoset.action_history:
            if item['action']: played_cards.extend(item['action'])
            
        unseen_cards = set(range(108)) - set(infoset.my_hand) - set(played_cards)
        unseen_feat = cards2array(list(unseen_cards)) # 108维
        
        level_feat = np.zeros(13, dtype=np.float32)
        if 0 <= cur_level_idx < 13: level_feat[cur_level_idx] = 1.0
        
        right, teammate, left = (me + 1) % 4, (me + 2) % 4, (me + 3) % 4
        cards_num_feat = np.zeros(84, dtype=np.float32)
        for i, pid in enumerate([right, teammate, left]):
            num = min(infoset.others_num.get(pid, 0), 27)
            cards_num_feat[i * 28 + num] = 1.0
            
        context_feat = np.concatenate([unseen_feat, level_feat, cards_num_feat]) # 108+13+84 = 205维

        # ==========================================
        # 2. 计算 History Features (最近 15 手牌滑动窗口)
        # ==========================================
        history_feats = np.zeros((MAX_HISTORY, 112), dtype=np.float32) # [15, 112]
        history_mask = np.zeros(MAX_HISTORY, dtype=np.float32)         # [15]
        
        recent_history = infoset.action_history[-MAX_HISTORY:]
        
        for i, item in enumerate(recent_history):
            pid, action = item['player_id'], item['action']
            rel_pos = (pid - me) % 4
            pos_feat = np.zeros(4, dtype=np.float32)
            pos_feat[rel_pos] = 1.0
            act_feat = cards2array(action)
            
            history_feats[i] = np.concatenate([pos_feat, act_feat])
            history_mask[i] = 1.0

        # ==========================================
        # 3. 提取所有合法动作，拼接 Query Batch
        # ==========================================
        last_move_info = None
        if infoset.last_move and infoset.last_pid != me:
            last_move_info = MoveDetector.get_move_type(infoset.last_move, cur_level_idx)
            
        generator = MoveGenerator(infoset.my_hand, cur_level_idx)
        legal_actions = generator.get_legal_actions(last_move_info)
        
        num_actions = len(legal_actions)
        my_hand_feat = cards2array(infoset.my_hand)
        
        # 组装网络所需的 Query 张量 (手牌 + 动作) -> 216 维
        query_batch = np.zeros((num_actions, 216), dtype=np.float32)
        for i, action in enumerate(legal_actions):
            act_feat = cards2array(action)
            query_batch[i] = np.concatenate([my_hand_feat, act_feat])
            
        # 根据动作数量进行矩阵平铺 (Tile)
        context_batch = np.tile(context_feat, (num_actions, 1))           # [Num_Actions, 205]
        history_batch = np.tile(history_feats, (num_actions, 1, 1))       # [Num_Actions, 15, 112]
        history_mask_batch = np.tile(history_mask, (num_actions, 1))      # [Num_Actions, 15]

        return {
            'x_batch': {
                'query': query_batch,         
                'context': context_batch,     
                'history': history_batch,     
                'history_mask': history_mask_batch 
            },
            'legal_actions': legal_actions,
            'player_id': me,
            'infoset': infoset
        }