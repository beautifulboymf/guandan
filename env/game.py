import random
from .utils import format_hand, get_card_name
from .settings import *
from .move_detector import MoveDetector


class InfoSet:
    """玩家视角的观察数据"""
    def __init__(self, player_id):
        self.player_id = player_id
        self.my_hand = []      # 我的手牌ID
        self.others_num = {}   # 其他人剩余牌数
        self.last_move = None  # 上一家的出牌
        self.last_pid = -1     # 谁出的上一手牌
        self.cur_level = 2     # 当前打几

class GameEnv:
    def __init__(self):
        self.num_players = 4
        self.deck = list(range(108)) # 0-107
        self.reset()

    def reset(self, current_level=2):
        self.cur_level = current_level
        self.game_over = False
        self.winner_order = [] # 记录谁先出完：[p1, p3...]
        
        # 洗牌
        random.shuffle(self.deck)
        
        # 发牌 (每人27张)
        self.players_hand = {i: [] for i in range(4)}
        for i in range(4):
            self.players_hand[i] = sorted(self.deck[i*27 : (i+1)*27])

        # 随机指定首出 (实际掼蛋有进贡规则决定)
        self.current_player = 0 
        self.last_move = [] 
        self.last_pid = -1 # 没人出牌
        self.history = []

        return self.get_infoset()

    def get_infoset(self):
        """生成当前玩家的观察状态"""
        info = InfoSet(self.current_player)
        info.my_hand = self.players_hand[self.current_player]
        info.others_num = {i: len(self.players_hand[i]) for i in range(4) if i != self.current_player}
        info.last_move = self.last_move
        info.last_pid = self.last_pid
        info.cur_level = self.cur_level
        return info

    def step(self, action):
        """
        action: list of card_ids
        """
        player = self.current_player
        cur_level_idx = self.cur_level - 2 # 2->0, 3->1 ...
        
        # --- 1. 验证是否为合法牌型 ---
        move_info = MoveDetector.get_move_type(action, cur_level_idx)
        
        if len(action) > 0 and move_info is None:
             print(f"非法牌型: {action}")
             # 返回错误，不做任何状态变更
             # 这里为了简单，如果非法，强制变为 PASS，或者抛异常让 UI 处理
             # 为了 human_test 体验，我们返回一个标记让 UI 提示重输
             return self.get_infoset(), 0, False, {'error': 'Invalid Move Type'}

        # --- 2. 验证是否能管住上家 ---
        if len(action) > 0:
            if self.last_move and self.last_pid != player:
                # 必须管住上家
                last_move_info = MoveDetector.get_move_type(self.last_move, cur_level_idx)
                if not self._can_beat(move_info, last_move_info):
                    print("打不过上家！")
                    return self.get_infoset(), 0, False, {'error': 'Cannot Beat Last Move'}
            
            # 验证通过，更新状态
            self.last_move = action
            self.last_pid = player
            for card in action:
                self.players_hand[player].remove(card)
            print(f"玩家 {player} 出牌: {move_info['type']} (Rank {move_info['rank']})")
        else:
            # PASS 逻辑
            if self.last_move and self.last_pid == player:
                # 如果是自己首出，不能PASS
                return self.get_infoset(), 0, False, {'error': 'Cannot Pass on First Move'}
            print(f"玩家 {player} PASS")

        # ... (后续的出完牌判定、轮转逻辑保持不变) ...
        
        # (最后加上 return)
        return self.get_infoset(), 0, self.game_over, {}

    def _can_beat(self, cur_move, last_move):
        """比较牌力逻辑"""
        t1, r1 = cur_move['type'], cur_move['rank']
        t2, r2 = last_move['type'], last_move['rank']
        
        # 1. 四大天王最大
        if t1 == TYPE_KING_BOMB: return True
        if t2 == TYPE_KING_BOMB: return False
        
        # 2. 炸弹比较 (含同花顺)
        # 炸弹等级：同花顺 > 5炸 > 4炸
        # 掼蛋中：6炸 > 同花顺 > 5炸 > 4炸 (具体规则有变种，这里按 6炸>同花顺>5炸)
        def get_bomb_score(m_type, count, rank):
            # 给炸弹打分，用于跨类型比较
            if m_type == TYPE_STRAIGHT_FLUSH: return 5.5 # 介于5炸和6炸之间
            if m_type == TYPE_BOMB: return count
            return 0 # 不是炸弹

        score1 = get_bomb_score(t1, cur_move['count'], r1)
        score2 = get_bomb_score(t2, last_move.get('count',0), r2) # last_move count需要 detector 返回
        
        # 如果都是炸弹 (分值 > 0)
        if score1 > 0 or score2 > 0:
            if score1 > score2: return True
            if score1 < score2: return False
            # 同等级炸弹，比点数
            return r1 > r2
            
        # 3. 普通牌型比较
        # 必须同类型、同张数 (MoveDetector 已经通过类型隐含了张数检查，如同是顺子都是5张)
        if t1 != t2: return False
        if cur_move['count'] != last_move['count']: return False
        
        return r1 > r2