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
        self.last_move = []    # 上一家的出牌
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
        self.winner_order = [] # 记录出完牌的顺序
        
        # 洗牌
        random.shuffle(self.deck)
        
        # 发牌 (每人27张)
        self.players_hand = {i: [] for i in range(4)}
        for i in range(4):
            self.players_hand[i] = sorted(self.deck[i*27 : (i+1)*27])

        # 状态初始化
        self.current_player = 0 
        self.last_move = []    # 桌面上的最后一手牌
        self.last_pid = -1     # 最后一手牌是谁出的
        self.pass_count = 0    # 连续 PASS 的次数
        
        # 记录每队的级数 (用于结算) - 简化起见这里只存储当前级
        # 实际项目中应由外部控制级数流转

        return self.get_infoset()

    def get_infoset(self):
        """生成当前玩家的观察状态"""
        info = InfoSet(self.current_player)
        info.my_hand = self.players_hand[self.current_player]
        # 获取其他人在场玩家的手牌数
        info.others_num = {i: len(self.players_hand[i]) for i in range(4) if i != self.current_player}
        info.last_move = self.last_move
        info.last_pid = self.last_pid
        info.cur_level = self.cur_level
        return info

    def step(self, action):
        """
        执行动作
        action: list of card_ids (空列表代表PASS)
        """
        player = self.current_player
        cur_level_idx = self.cur_level - 2 
        
        # === 1. 强制出牌检查 ===
        # 如果桌面为空（获得出牌权），或者是自己出的最后一手（理论上会被清空逻辑覆盖，但也做防御），决不允许 PASS
        if not self.last_move or self.last_pid == player:
            if len(action) == 0:
                return self.get_infoset(), 0, False, {'error': '必须出牌 (Cannot Pass)'}

        # === 2. 动作处理 ===
        if len(action) > 0:
            # --- 出牌逻辑 ---
            # A. 合法性检查
            move_info = MoveDetector.get_move_type(action, cur_level_idx)
            if move_info is None:
                return self.get_infoset(), 0, False, {'error': '非法牌型 (Invalid Move)'}

            # B. 压制检查
            if self.last_move and self.last_pid != player:
                last_move_info = MoveDetector.get_move_type(self.last_move, cur_level_idx)
                if not self._can_beat(move_info, last_move_info):
                    return self.get_infoset(), 0, False, {'error': '管不住上家 (Too Small)'}

            # C. 执行出牌
            self.last_move = action
            self.last_pid = player
            self.pass_count = 0  # 有人出牌，PASS计数清零
            
            for card in action:
                self.players_hand[player].remove(card)
            print(f"玩家 {player} 出牌: {move_info['type']} (Rank {move_info['rank']})")

        else:
            # --- PASS 逻辑 ---
            self.pass_count += 1
            print(f"玩家 {player} PASS")

        # === 3. 检查当前玩家是否出完 ===
        if len(self.players_hand[player]) == 0:
            if player not in self.winner_order:
                self.winner_order.append(player)
                print(f"*** 玩家 {player} 出完了! (第 {len(self.winner_order)} 名) ***")

        # === 4. 胜负判定 (Game Over Check) ===
        # 只要有3个人出完，或者同队的两人都出完（双上），游戏结束
        # 简化判定：winner_order 达到 3 人，或者 双上
        is_over = False
        if len(self.winner_order) >= 3:
            is_over = True
        elif len(self.winner_order) == 2:
            # 检查是不是双上 (0和2是一队，1和3是一队)
            p1, p2 = self.winner_order
            if (p1 % 2) == (p2 % 2):
                is_over = True # 双上，不用等那俩倒霉蛋了

        if is_over:
            self.game_over = True
            result = self._calculate_result()
            return self.get_infoset(), 0, True, {'result': result}

        # === 5. 轮转与接风逻辑 (Next Player Logic) ===
        
        # 计算在场的活跃玩家数量 (没出完牌的人)
        active_players = [i for i in range(4) if len(self.players_hand[i]) > 0]
        num_active = len(active_players)
        
        # 寻找下一个出牌者（普通轮转）
        next_p = (player + 1) % 4
        while next_p not in active_players:
            next_p = (next_p + 1) % 4
        
        # 检查是否由于连续 PASS 导致需要清空桌面（接风/获得球权）
        # 触发条件：连续 PASS 的人数 == 当前在场人数 - 1
        # (例如剩4人，3人PASS，第4人获得球权；剩2人，1人PASS，另1人获得)
        # 注意：如果是刚刚出牌的那个人自己出完了，pass_count 是 0，逻辑在下面处理
        
        should_clear_table = False
        winner_next_p = next_p # 默认下家

        if self.pass_count >= num_active - 1:
            # --- 一圈没人要，准备清空桌面 ---
            print(f"--- 一圈没人要 (Pass Count: {self.pass_count}) ---")
            should_clear_table = True
            
            # 关键：谁来接风？
            # 正常情况：最后出牌的人 (last_pid) 获得球权
            if len(self.players_hand[self.last_pid]) > 0:
                winner_next_p = self.last_pid
                print(f"-> 玩家 {winner_next_p} 获得出牌权")
            else:
                # 特殊情况：最后出牌的人已经出完走了 (头游/二游) -> 【接风规则】
                # 规则：优先给对家接风
                teammate = (self.last_pid + 2) % 4
                if len(self.players_hand[teammate]) > 0:
                    winner_next_p = teammate
                    print(f"-> 最大牌玩家已走，队友 {winner_next_p} 接风")
                else:
                    # 如果对家也走了（极其罕见，通常双上就结束了，除非是特殊规则），给下家
                    # 下家就是目前算出来的 next_p
                    winner_next_p = next_p
                    print(f"-> 最大牌玩家及队友均已走，下家 {winner_next_p} 接风")
            
            # 重置 PASS 计数
            self.pass_count = 0

        # 如果刚才出牌的人出完了（且不是接风情况，即有人刚出完牌，还没轮一圈），
        # 此时 pass_count=0，next_p 正常流转给下家，桌面保持（下家必须管）
        # 除非它是最后一张且最大... 那个逻辑由上面的 pass_count 累积后的下一轮处理
        
        if should_clear_table:
            self.last_move = [] # 清空桌面
            self.current_player = winner_next_p
        else:
            self.current_player = next_p

        return self.get_infoset(), 0, False, {}

    def _can_beat(self, cur_move, last_move):
        """比较牌力逻辑"""
        t1, r1 = cur_move['type'], cur_move['rank']
        t2, r2 = last_move['type'], last_move['rank']
        
        if t1 == TYPE_KING_BOMB: return True
        if t2 == TYPE_KING_BOMB: return False
        
        def get_bomb_score(m_type, count, rank):
            if m_type == TYPE_STRAIGHT_FLUSH: return 5.5
            if m_type == TYPE_BOMB: return count
            return 0

        last_count = last_move.get('count', 0)
        score1 = get_bomb_score(t1, cur_move['count'], r1)
        score2 = get_bomb_score(t2, last_count, r2) 
        
        if score1 > 0 or score2 > 0:
            if score1 > score2: return True
            if score1 < score2: return False
            return r1 > r2
            
        if t1 != t2: return False
        if cur_move['count'] != last_count: return False
        return r1 > r2

    def _calculate_result(self):
        """结算胜负与升级"""
        # winner_order 存储了出完牌的顺序，例如 [0, 2, 1] (0和2一队，1和3一队)
        # 还没出完的人算最后一名
        
        team_A = [0, 2] # 玩家 0, 2
        team_B = [1, 3] # 玩家 1, 3
        
        first = self.winner_order[0]
        
        # 判断是哪队拿了头游
        win_team_idx = 'A' if first in team_A else 'B'
        
        # 查找队友的名次
        teammate = (first + 2) % 4
        teammate_rank = -1 # 没出完
        if teammate in self.winner_order:
            # index是从0开始的，所以名次是 index + 1
            teammate_rank = self.winner_order.index(teammate) + 1
        else:
            teammate_rank = 4 # 最后一名
            
        result_str = ""
        level_up = 0
        
        if teammate_rank == 2:
            result_str = f"Team {win_team_idx} 双上 (头游+二游)"
            level_up = 3 # 升3级
        elif teammate_rank == 3:
            result_str = f"Team {win_team_idx} 单上 (头游+三游)"
            level_up = 2 # 升2级
        else: # teammate_rank == 4
            result_str = f"Team {win_team_idx} 平局/单上 (头游+末游)"
            level_up = 1 # 升1级
            
        return {
            'winner': win_team_idx,
            'desc': result_str,
            'level_up': level_up,
            'order': self.winner_order
        }