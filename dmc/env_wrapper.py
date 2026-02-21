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
from env.settings import TYPE_BOMB, TYPE_STRAIGHT_FLUSH, TYPE_KING_BOMB 

# 【优化 1】扩充历史记忆，足以覆盖一整局
MAX_HISTORY = 40 

def cards2array(card_ids):
    vec = np.zeros(108, dtype=np.float32)
    if not card_ids: return vec
        
    from collections import Counter
    type_ids = []
    for cid in card_ids:
        if cid < 104: type_ids.append(cid % 52)
        elif cid in [104, 105]: type_ids.append(52) 
        else: type_ids.append(53)                   
            
    for tid, count in Counter(type_ids).items():
        if count >= 1: vec[tid] = 1.0
        if count == 2: vec[tid + 54] = 1.0
    return vec

class GuandanEnvWrapper:
    def __init__(self):
        self.env = GameEnv()

    def reset(self, current_level=2):
        infoset = self.env.reset(current_level)
        return self._get_obs(infoset)

    def step(self, action):
        next_infoset, reward, done, result_info = self.env.step(action)
        obs = self._get_obs(next_infoset) if not done else None
        return obs, reward, done, result_info

    def _prune_actions(self, legal_actions, infoset):
        if len(legal_actions) <= 1:
            return legal_actions 
            
        pruned = []
        my_pid = infoset.player_id
        last_pid = infoset.last_pid
        teammate_pid = (my_pid + 2) % 4
        my_hand_len = len(infoset.my_hand)
        cur_level_idx = infoset.cur_level - 2
        
        is_free_play = (not infoset.last_move) or (last_pid == my_pid)
        
        for action in legal_actions:
            if not action: 
                pruned.append(action)
                continue
                
            move_info = MoveDetector.get_move_type(action, cur_level_idx)
            if not move_info: continue
                
            m_type = move_info['type']
            is_bomb = m_type in [TYPE_BOMB, TYPE_STRAIGHT_FLUSH, TYPE_KING_BOMB]
            
            if is_free_play and is_bomb and len(action) != my_hand_len:
                continue
            if last_pid == teammate_pid and is_bomb and my_hand_len > 5:
                continue
            pruned.append(action)
            
        return pruned if pruned else legal_actions

    def _get_obs(self, infoset):
        me = infoset.player_id
        cur_level_idx = infoset.cur_level - 2
        
        # 1. Context Feature 
        played_cards = []
        for item in infoset.action_history:
            if item['action']: played_cards.extend(item['action'])
            
        unseen_cards = set(range(108)) - set(infoset.my_hand) - set(played_cards)
        unseen_feat = cards2array(list(unseen_cards)) 
        
        level_feat = np.zeros(13, dtype=np.float32)
        if 0 <= cur_level_idx < 13: level_feat[cur_level_idx] = 1.0
        
        # 【优化 2】极其精简且无损的信息量：直接传入 3 个整数 (0~27)
        right, teammate, left = (me + 1) % 4, (me + 2) % 4, (me + 3) % 4
        cards_num_feat = np.zeros(3, dtype=np.float32)
        for i, pid in enumerate([right, teammate, left]):
            cards_num_feat[i] = float(min(infoset.others_num.get(pid, 0), 27))
            
        # 此时 context_feat 维度: 108 + 13 + 3 = 124 维
        context_feat = np.concatenate([unseen_feat, level_feat, cards_num_feat]) 

        # 2. History Features 
        history_feats = np.zeros((MAX_HISTORY, 112), dtype=np.float32) 
        history_mask = np.zeros(MAX_HISTORY, dtype=np.float32)         
        
        recent_history = infoset.action_history[-MAX_HISTORY:]
        
        for i, item in enumerate(recent_history):
            pid, action = item['player_id'], item['action']
            rel_pos = (pid - me) % 4
            pos_feat = np.zeros(4, dtype=np.float32)
            pos_feat[rel_pos] = 1.0
            act_feat = cards2array(action)
            
            # 这里依然保持 112 维拼接，我们在 PyTorch 模型内部去解剖它
            history_feats[i] = np.concatenate([pos_feat, act_feat])
            history_mask[i] = 1.0

        # 3. 动作提取与组装
        last_move_info = None
        if infoset.last_move and infoset.last_pid != me:
            last_move_info = MoveDetector.get_move_type(infoset.last_move, cur_level_idx)
            
        generator = MoveGenerator(infoset.my_hand, cur_level_idx)
        raw_legal_actions = generator.get_legal_actions(last_move_info)
        legal_actions = self._prune_actions(raw_legal_actions, infoset)
        num_actions = len(legal_actions)
        
        my_hand_feat = cards2array(infoset.my_hand)
        
        query_batch = np.zeros((num_actions, 216), dtype=np.float32)
        for i, action in enumerate(legal_actions):
            act_feat = cards2array(action)
            query_batch[i] = np.concatenate([my_hand_feat, act_feat])
            
        context_batch = np.tile(context_feat, (num_actions, 1))           
        history_batch = np.tile(history_feats, (num_actions, 1, 1))       
        history_mask_batch = np.tile(history_mask, (num_actions, 1))      

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