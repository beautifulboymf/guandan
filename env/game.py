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

        # --- 3. 检查是否出完 (胜负判定) ---
        if len(self.players_hand[player]) == 0:
            if player not in self.winner_order:
                self.winner_order.append(player)
                print(f"*** 玩家 {player} 出完了! (第 {len(self.winner_order)} 名) ***")

        # 检查游戏结束 (只要3个人出完就结束)
        if len(self.winner_order) >= 3:
            self.game_over = True
            return self.get_infoset(), 0, True, {}

        # --- 4. 轮转到下一位 (关键逻辑) ---
        next_p = (player + 1) % 4
        while next_p in self.winner_order:
            if len(self.winner_order) == 4: break 
            next_p = (next_p + 1) % 4
        
        # 检查是否获得“接风”或新一轮出牌权
        # 如果 last_pid 也是 next_p (说明其他人都要不起，转回自己了)，或者 last_pid 已经赢了走了
        # 这里简化逻辑：如果上一手牌的打出者就是下家自己，或者上一手牌打出者已经离场（这种其实需要更复杂的接风逻辑）
        # 我们暂时只处理：转了一圈没人要，轮到自己 -> 清空 last_move
        if self.last_pid == next_p:
            print(f"--- 没人要，玩家 {next_p} 获得出牌权 ---")
            self.last_move = []
        
        # 更新当前玩家
        self.current_player = next_p
        
        return self.get_infoset(), 0, self.game_over, {}
    
    def _can_beat(self, cur_move, last_move):
        """比较牌力逻辑：判断 cur_move 是否能管住 last_move"""
        t1, r1 = cur_move['type'], cur_move['rank']
        t2, r2 = last_move['type'], last_move['rank']
        
        # 1. 四大天王最大
        if t1 == TYPE_KING_BOMB: return True
        if t2 == TYPE_KING_BOMB: return False
        
        # 2. 炸弹比较 (含同花顺)
        # 掼蛋炸弹等级一般规则：6炸及以上 > 同花顺 > 5炸 > 4炸
        # (这里为了简化，我们暂时使用一个分数来代表炸弹等级)
        def get_bomb_score(m_type, count, rank):
            # 给炸弹打分，用于跨类型比较
            if m_type == TYPE_STRAIGHT_FLUSH: return 5.5 # 设定同花顺比5炸大，比6炸小
            if m_type == TYPE_BOMB: return count # 4炸得4分，5炸得5分...
            if m_type == TYPE_KING_BOMB: return 100
            return 0 # 不是炸弹

        # 获取 last_move 的 count，如果没有则默认为 0
        last_count = last_move.get('count', 0)
        
        score1 = get_bomb_score(t1, cur_move['count'], r1)
        score2 = get_bomb_score(t2, last_count, r2) 
        
        # 如果其中一方是炸弹 (分值 > 0)
        if score1 > 0 or score2 > 0:
            if score1 > score2: return True # 炸弹等级高这就赢
            if score1 < score2: return False # 炸弹等级低这就输
            # 同等级炸弹，比点数 (例如都是4炸，AAAA > KKKK)
            return r1 > r2
            
        # 3. 普通牌型比较
        # 必须同类型
        if t1 != t2: return False
        # 必须同张数 (例如对子只能管对子，三带二只能管三带二)
        if cur_move['count'] != last_count: return False
        # 比主牌点数
        return r1 > r2