import numpy as np
import os
import sys
from collections import Counter

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.game import GameEnv
from env.move_generator import MoveGenerator
from env.move_detector import MoveDetector
from env.utils import get_logical_rank, is_wild_card

# 记忆翻倍，128手绝对记忆，完全覆盖一局掼蛋的极限长度
MAX_HISTORY = 128 

def cards2array(card_ids):
    vec = np.zeros(108, dtype=np.float32)
    if not card_ids: return vec
    type_ids = []
    for cid in card_ids:
        if cid < 104: type_ids.append(cid % 52)
        elif cid in [104, 105]: type_ids.append(52) 
        else: type_ids.append(53)                   
    for tid, count in Counter(type_ids).items():
        if count >= 1: vec[tid] = 1.0
        if count == 2: vec[tid + 54] = 1.0
    return vec

def get_macro_features(hand_ids, cur_level_idx):
    """
    【上帝视角特征】：宏观提取手牌的非重叠（Disjoint）组合潜力
    这里不包含逢人配的枚举，而是提取“纯自然牌”的骨架！
    神经网络会根据这里的自然骨架特征 + 逢人配数量，自动推断出最终的牌力。
    """
    features = np.zeros(24, dtype=np.float32)
    if not hand_ids: return features
    
    features[21] = len(hand_ids) # 维度21: 总牌数
    
    wilds, jokers, normals = [], [], []
    for cid in hand_ids:
        if is_wild_card(cid, cur_level_idx): wilds.append(cid)
        elif cid >= 104: jokers.append(cid)
        else: normals.append(cid)
            
    features[13] = len(wilds)  # 维度13: 逢人配数量
    features[14] = len(jokers) # 维度14: 大小王数量
    if len(jokers) == 4:
        features[0] = 1.0 # 维度0: 四大天王 (King Bomb)
        jokers = []
        
    # 统计绝对大牌的威慑力
    for cid in hand_ids:
        lr = get_logical_rank(cid, cur_level_idx)
        if lr == 15: features[15] += 1.0 # 维度15: 级牌数量
        elif lr == 14: features[16] += 1.0 # A
        elif lr == 13: features[17] += 1.0 # K
        elif lr == 12: features[18] += 1.0 # Q
        elif lr == 11: features[19] += 1.0 # J
        elif lr == 10: features[20] += 1.0 # 10
    
    # ==========================================
    # 核心算法：贪心消耗式提取不相交组合 (Disjoint Set)
    # 优先级：顺子 -> 钢板 -> 三连对 -> 炸弹 -> 三张 -> 对子 -> 单张
    # 每提取出一个组合，就从 rank_counts 里彻底删掉它！
    # ==========================================
    rank_counts = Counter()
    for cid in normals:
        nat_r = (cid % 13) + 2
        rank_counts[nat_r] += 1
        
    # 1. 顺子 (最长优先，从 12 连顺往下找，直到 5 连顺)
    num_straights = 0
    for l in range(12, 4, -1):
        for start in range(2, 15 - l + 1):
            while all(rank_counts[r] >= 1 for r in range(start, start + l)):
                num_straights += 1
                for r in range(start, start + l): rank_counts[r] -= 1
    features[9] = num_straights # 维度9: 顺子数
    
    # 2. 钢板 (333444)
    num_plates = 0
    for start in range(2, 14):
        while rank_counts[start] >= 3 and rank_counts[start+1] >= 3:
            num_plates += 1
            rank_counts[start] -= 3
            rank_counts[start+1] -= 3
    features[7] = num_plates # 维度7: 钢板数
    
    # 3. 三连对 (334455)
    num_tubes = 0
    for start in range(2, 13):
        while rank_counts[start] >= 2 and rank_counts[start+1] >= 2 and rank_counts[start+2] >= 2:
            num_tubes += 1
            rank_counts[start] -= 2
            rank_counts[start+1] -= 2
            rank_counts[start+2] -= 2
    features[8] = num_tubes # 维度8: 三连对数
    
    # 4. 炸弹 (此时剩下的全是独立的散牌聚簇了)
    for r, c in list(rank_counts.items()):
        if c >= 4:
            length = min(c, 8)
            features[2 + (8 - length)] += 1.0 # 维度2-6分别对应8/7/6/5/4张的炸弹
            rank_counts[r] -= c # 把炸弹彻底抽离
            
    # 5. 三张
    num_triples = 0
    for r, c in list(rank_counts.items()):
        if c == 3:
            num_triples += 1
            rank_counts[r] -= 3
    features[10] = num_triples # 维度10: 净三张数
    
    # 6. 对子
    num_pairs = 0
    for r, c in list(rank_counts.items()):
        if c == 2:
            num_pairs += 1
            rank_counts[r] -= 2
    features[11] = num_pairs # 维度11: 净对子数
    
    # 7. 单张 (经过前面层层剥削，剩下的全是纯单牌)
    features[12] = sum(rank_counts.values()) + len(jokers) # 维度12: 绝对单牌数
    
    return features


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

    # 彻底废除“不准拆炸弹”、“不准炸队友”的封印！自由度全开！
    def _prune_actions(self, legal_actions, infoset):
        return legal_actions

    def _get_obs(self, infoset):
        me = infoset.player_id
        cur_level_idx = infoset.cur_level - 2
        
        played_cards = []
        for item in infoset.action_history:
            if item['action']: played_cards.extend(item['action'])
            
        unseen_cards = set(range(108)) - set(infoset.my_hand) - set(played_cards)
        unseen_feat = cards2array(list(unseen_cards)) 
        
        safe_level_idx = max(0, min(12, cur_level_idx))
        level_feat = np.array([safe_level_idx], dtype=np.float32) 
        
        right, teammate, left = (me + 1) % 4, (me + 2) % 4, (me + 3) % 4
        cards_num_feat = np.zeros(3, dtype=np.float32) 
        for i, pid in enumerate([right, teammate, left]):
            cards_num_feat[i] = float(min(infoset.others_num.get(pid, 0), 27))
            
        # ==========================================
        # [史诗级增强 1] 融入 24维宏观骨架特征
        # Context 维度从 112 暴涨至 136，AI 瞬间有了大局观
        # ==========================================
        macro_feat = get_macro_features(infoset.my_hand, cur_level_idx)
        context_feat = np.concatenate([unseen_feat, level_feat, cards_num_feat, macro_feat]) 

        history_feats = np.zeros((MAX_HISTORY, 112), dtype=np.float32) 
        history_mask = np.zeros(MAX_HISTORY, dtype=np.float32)         
        
        recent_history = infoset.action_history[-MAX_HISTORY:]
        recent_history.reverse()
        
        for i, item in enumerate(recent_history):
            pid, action = item['player_id'], item['action']
            pos_feat = np.zeros(4, dtype=np.float32)
            pos_feat[(pid - me) % 4] = 1.0
            act_feat = cards2array(action)
            history_feats[i] = np.concatenate([pos_feat, act_feat])
            history_mask[i] = 1.0

        last_move_info = None
        if infoset.last_move and infoset.last_pid != me:
            last_move_info = MoveDetector.get_move_type(infoset.last_move, cur_level_idx)
            
        generator = MoveGenerator(infoset.my_hand, cur_level_idx)
        legal_actions = self._prune_actions(generator.get_legal_actions(last_move_info), infoset)
        num_actions = len(legal_actions)
        
        my_hand_feat = cards2array(infoset.my_hand)
        query_batch = np.zeros((num_actions, 216), dtype=np.float32)
        for i, action in enumerate(legal_actions):
            query_batch[i] = np.concatenate([my_hand_feat, cards2array(action)])
            
        # ==========================================
        # [史诗级增强 2] 提取“隐藏的上帝视角标签”
        # 我们偷看其他三个人的牌，作为后续预测头(Prediction Head)的监督信号
        # ==========================================
        hidden_right = cards2array(self.env.players_hand[right])
        hidden_teammate = cards2array(self.env.players_hand[teammate])
        hidden_left = cards2array(self.env.players_hand[left])
        hidden_labels = np.concatenate([hidden_right, hidden_teammate, hidden_left]) # 108 * 3 = 324 维
            
        return {
            'x_batch': {
                'query': query_batch,         
                'context': np.tile(context_feat, (num_actions, 1)),     
                'history': np.tile(history_feats, (num_actions, 1, 1)),     
                'history_mask': np.tile(history_mask, (num_actions, 1)),
                'hidden_labels': np.tile(hidden_labels, (num_actions, 1)) # 打包发送给 GPU
            },
            'legal_actions': legal_actions, 'player_id': me, 'infoset': infoset
        }