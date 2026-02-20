import itertools
from collections import defaultdict
import os
import sys

# === 解决 Python 绝对/相对导入路径报错的核心代码 ===
# 动态获取当前脚本的绝对路径，并推算出项目的根目录（guandan/）加入系统路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一使用从根目录开始的绝对导入
from env.settings import *
from env.utils import is_wild_card, get_logical_rank, format_hand, get_card_name
from env.move_detector import MoveDetector # [新增] 导入牌型识别器

class MoveGenerator:
    def __init__(self, hand_card_ids, cur_level_rank_idx):
        self.hand = hand_card_ids
        self.cur_level_idx = cur_level_rank_idx
        
        self.wild_cards = []
        self.normal_cards_by_rank = defaultdict(list)
        self.natural_cards_by_rank = defaultdict(list)
        self.natural_cards_by_suit_and_rank = {suit: defaultdict(list) for suit in range(4)}
        
        self.jokers = [] # [新增]：单独存放大小王，绝对不能和逢人配混用
        
        for cid in self.hand:
            if is_wild_card(cid, self.cur_level_idx):
                self.wild_cards.append(cid)
            elif cid >= 104:
                self.jokers.append(cid) # 大小王进小黑屋隔离
            else:
                # 只有 2~A 才进入 normal_cards，允许被逢人配替代
                logic_rank = get_logical_rank(cid, self.cur_level_idx)
                self.normal_cards_by_rank[logic_rank].append(cid)
                
                nat_rank = (cid % 13) + 2
                suit_idx = (cid % 52) // 13
                
                self.natural_cards_by_rank[nat_rank].append(cid)
                self.natural_cards_by_suit_and_rank[suit_idx][nat_rank].append(cid)
                
                if nat_rank == 14:
                    self.natural_cards_by_rank[1].append(cid)
                    self.natural_cards_by_suit_and_rank[suit_idx][1].append(cid)
    
    def _dedup_moves(self, moves):
        """
        [极其重要的 RL 优化]: 动作逻辑去重
        如果手牌有两张黑桃3 (S3_deck1, S3_deck2)，它们组成的对子/单张逻辑上是一致的。
        我们通过比较牌面名称来去重，极大压缩 RL 的 Action Space。
        """
        unique_moves = []
        seen = set()
        for move in moves:
            # 提取牌名并排序，例如 ('H3', 'S3')
            name_tuple = tuple(sorted([get_card_name(cid) for cid in move]))
            if name_tuple not in seen:
                seen.add(name_tuple)
                unique_moves.append(sorted(move))
        return unique_moves

    def gen_singles(self):
        """生成所有单张"""
        moves = [[cid] for cid in self.hand]
        return self._dedup_moves(moves)

    def gen_pairs(self):
        """生成所有对子"""
        moves = []
        # 1. 纯自然牌的对子
        for rank, ids in self.normal_cards_by_rank.items():
            if len(ids) >= 2:
                # itertools.combinations 会穷举列表里任意2张的组合
                for combo in itertools.combinations(ids, 2):
                    moves.append(list(combo))
                    
        # 2. 自然牌(1张) + 逢人配(1张)
        if len(self.wild_cards) >= 1:
            for rank, ids in self.normal_cards_by_rank.items():
                for combo in itertools.combinations(ids, 1):
                    for w_combo in itertools.combinations(self.wild_cards, 1):
                        moves.append(list(combo) + list(w_combo))
                        
        # 3. 逢人配(2张) 组成的对子
        if len(self.wild_cards) >= 2:
            for combo in itertools.combinations(self.wild_cards, 2):
                moves.append(list(combo))
                
        # 4. [新增] 纯天然的大小王对子
        sb_jokers = [cid for cid in self.jokers if cid in [104, 105]]
        hr_jokers = [cid for cid in self.jokers if cid in [106, 107]]
        if len(sb_jokers) == 2:
            moves.append(sb_jokers)
        if len(hr_jokers) == 2:
            moves.append(hr_jokers)
    
        return self._dedup_moves(moves)

    def gen_triples(self):
        """生成所有三张"""
        moves = []
        for rank, ids in self.normal_cards_by_rank.items():
            # 1. 纯自然牌三张
            if len(ids) >= 3:
                for combo in itertools.combinations(ids, 3):
                    moves.append(list(combo))
                    
            # 2. 自然牌(2张) + 逢人配(1张)
            if len(ids) >= 2 and len(self.wild_cards) >= 1:
                for combo in itertools.combinations(ids, 2):
                    for w_combo in itertools.combinations(self.wild_cards, 1):
                        moves.append(list(combo) + list(w_combo))
                        
            # 3. 自然牌(1张) + 逢人配(2张)
            if len(ids) >= 1 and len(self.wild_cards) >= 2:
                for combo in itertools.combinations(ids, 1):
                    for w_combo in itertools.combinations(self.wild_cards, 2):
                        moves.append(list(combo) + list(w_combo))
        
        # 掼蛋只有两副牌，最多2张逢人配，所以不可能出现3张逢人配组成的三张
        return self._dedup_moves(moves)
    
    def gen_triple_pairs(self):
        """生成所有三带二 (包含逢人配补位)"""
        moves = []
        
        # t_len: 组成三张部分所使用的自然牌数量 (1张, 2张, 3张)
        # p_len: 组成对子部分所使用的自然牌数量 (0张, 1张, 2张)
        for t_len in range(1, 4):
            for p_len in range(0, 3):
                # 计算需要的百搭牌总数
                w_needed = (3 - t_len) + (2 - p_len)
                if w_needed > len(self.wild_cards):
                    continue # 百搭牌不够，跳过
                
                for t_rank, t_cards in self.normal_cards_by_rank.items():
                    if len(t_cards) < t_len: continue
                    
                    # 如果 p_len == 0，说明对子完全由2张百搭牌构成 (如: 333 + 两个逢人配)
                    if p_len == 0:
                        for tc in itertools.combinations(t_cards, t_len):
                            for wc in itertools.combinations(self.wild_cards, w_needed):
                                moves.append(list(tc) + list(wc))
                        continue
                        
                    # p_len > 0 的情况
                    for p_rank, p_cards in self.normal_cards_by_rank.items():
                        if t_rank == p_rank: continue # 三带二的主牌和副牌点数必须不同
                        if len(p_cards) < p_len: continue
                        
                        for tc in itertools.combinations(t_cards, t_len):
                            for pc in itertools.combinations(p_cards, p_len):
                                for wc in itertools.combinations(self.wild_cards, w_needed):
                                    moves.append(list(tc) + list(pc) + list(wc))
                                    
        return self._dedup_moves(moves)

    def gen_straights(self):
        """生成所有普通顺子 (排除同花顺)"""
        moves = []
        
        # 顺子起点从 1 (即 A-2-3-4-5) 到 10 (即 10-J-Q-K-A)
        for start_rank in range(1, 11):
            sequence = list(range(start_rank, start_rank + 5))
            
            options = []
            missing_count = 0
            
            # 统计这个顺子我们在手牌中缺几张
            for r in sequence:
                cards = self.natural_cards_by_rank.get(r, [])
                if not cards:
                    missing_count += 1
                else:
                    options.append(cards)
            
            # 如果缺的牌少于等于手里的百搭牌，就能凑出顺子
            if missing_count <= len(self.wild_cards):
                # itertools.product 生成花色的笛卡尔积 (比如黑桃3和红桃3，会分出两条路径)
                for prod in itertools.product(*options):
                    
                    # 检查是否为同花顺：如果挑选出的自然牌全是一个花色，说明是同花顺
                    suits = set()
                    for cid in prod:
                        suit_idx = (cid % 52) // 13
                        suits.add(suit_idx)
                        
                    if len(suits) == 1:
                        # 纯同花色的情况交给 gen_straight_flushes 处理，这里剔除
                        continue
                        
                    # 拼上百搭牌
                    for wc in itertools.combinations(self.wild_cards, missing_count):
                        moves.append(list(prod) + list(wc))
                        
        return self._dedup_moves(moves)
    
    def _get_rank_choices(self, rank, target_count):
        """
        核心匹配器：为指定点数挑选自然牌组合
        target_count: 连对是2，钢板是3
        """
        choices = []
        cards = self.natural_cards_by_rank.get(rank, [])
        max_wilds = len(self.wild_cards)
        
        # 最少需要挑几张自然牌 (剩下的用百搭牌补)
        min_pick = max(0, target_count - max_wilds)
        # 最多能挑几张
        max_pick = min(len(cards), target_count)
        
        for k in range(min_pick, max_pick + 1):
            for combo in itertools.combinations(cards, k):
                choices.append(list(combo))
        return choices

    def gen_tubes(self):
        """生成三连对 (如 334455)"""
        moves = []
        # 起点从 1(A23) 到 12(QQKKAA)
        for start_rank in range(2, 13):
            seq = [start_rank, start_rank+1, start_rank+2]
            
            # 获取这3个点数的所有可能挑法
            all_choices = [self._get_rank_choices(r, 2) for r in seq]
            
            # 如果某个点数连一张牌+百搭牌都凑不出，直接断掉
            if any(not choices for choices in all_choices):
                continue
                
            # 笛卡尔积交叉组合
            for prod in itertools.product(*all_choices):
                combined = []
                for choice in prod:
                    combined.extend(choice)
                    
                missing = 6 - len(combined)
                if missing <= len(self.wild_cards):
                    for wc in itertools.combinations(self.wild_cards, missing):
                        moves.append(combined + list(wc))
                        
        return self._dedup_moves(moves)

    def gen_plates(self):
        """生成钢板 (两连三张，如 333444)"""
        moves = []
        # 起点从 1(AA22) 到 13(KKAA)
        for start_rank in range(2, 14):
            seq = [start_rank, start_rank+1]
            all_choices = [self._get_rank_choices(r, 3) for r in seq]
            
            if any(not choices for choices in all_choices):
                continue
                
            for prod in itertools.product(*all_choices):
                combined = []
                for choice in prod:
                    combined.extend(choice)
                    
                missing = 6 - len(combined)
                if missing <= len(self.wild_cards):
                    for wc in itertools.combinations(self.wild_cards, missing):
                        moves.append(combined + list(wc))
                        
        return self._dedup_moves(moves)

    def gen_straight_flushes(self):
        """生成同花顺 (包含逢人配替代)"""
        moves = []
        
        for suit in range(4):
            for start_rank in range(1, 11): # 起点 1(A2345) 到 10(10JQKA)
                seq = list(range(start_rank, start_rank + 5))
                options = []
                missing_count = 0
                
                # 同花顺的检查更严格，必须在这个 suit 下找
                for r in seq:
                    cards = self.natural_cards_by_suit_and_rank[suit].get(r, [])
                    if not cards:
                        missing_count += 1
                    else:
                        options.append(cards)
                        
                if missing_count <= len(self.wild_cards):
                    for prod in itertools.product(*options):
                        for wc in itertools.combinations(self.wild_cards, missing_count):
                            moves.append(list(prod) + list(wc))
                            
        return self._dedup_moves(moves)
    
    def gen_bombs(self):
        """生成普通炸弹 (4-8张，包含逢人配替代)"""
        moves = []
        
        # 遍历所有逻辑点数 (包括级牌 15)
        for rank, cards in self.normal_cards_by_rank.items():
            # 掼蛋的炸弹长度从 4 到 8
            for length in range(4, 9):
                max_natural_available = len(cards)
                max_wild_available = len(self.wild_cards)
                
                # 最少需要的自然牌数量 (剩下的由百搭牌补)
                # 至少需要 1 张自然牌来定义炸弹的点数
                min_pick = max(1, length - max_wild_available)
                max_pick = min(length, max_natural_available)
                
                if min_pick <= max_pick:
                    for k in range(min_pick, max_pick + 1):
                        w_needed = length - k
                        for n_combo in itertools.combinations(cards, k):
                            for w_combo in itertools.combinations(self.wild_cards, w_needed):
                                moves.append(list(n_combo) + list(w_combo))
                                
        return self._dedup_moves(moves)

    def gen_king_bomb(self):
        """生成四大天王 (四个王)"""
        # 大小王的 ID 是 >= 104 的
        jokers = [cid for cid in self.hand if cid >= 104]
        if len(jokers) == 4:
            return self._dedup_moves([jokers])
        return []
    
    @staticmethod
    def _can_beat(cur_move_info, last_move_info):
        """
        独立的牌力比较逻辑：判断 cur_move_info 能否压制 last_move_info
        """
        t1, r1 = cur_move_info['type'], cur_move_info['rank']
        t2, r2 = last_move_info['type'], last_move_info['rank']
        
        if t1 == TYPE_KING_BOMB: return True
        if t2 == TYPE_KING_BOMB: return False
        
        def get_bomb_score(m_type, count, rank):
            if m_type == TYPE_STRAIGHT_FLUSH: return 5.5
            if m_type == TYPE_BOMB: return count
            return 0

        last_count = last_move_info.get('count', 0)
        score1 = get_bomb_score(t1, cur_move_info['count'], r1)
        score2 = get_bomb_score(t2, last_count, r2) 
        
        # 炸弹之间的跨维打击
        if score1 > 0 or score2 > 0:
            if score1 > score2: return True
            if score1 < score2: return False
            return r1 > r2
            
        # 普通牌型比较：必须类型和张数相同，且点数更大
        if t1 != t2: return False
        if cur_move_info['count'] != last_count: return False
        return r1 > r2

    def _filter_greater(self, candidate_moves, last_move_info):
        """核心过滤器：定向筛出足以压制上家的组合"""
        valid_moves = []
        for move in candidate_moves:
            # 识别当前候选组合的牌型特征
            cur_move_info = MoveDetector.get_move_type(move, self.cur_level_idx)
            
            # 如果是合法牌型，并且能管住上家，则保留
            if cur_move_info and self._can_beat(cur_move_info, last_move_info):
                valid_moves.append(move)
        return valid_moves

    def get_legal_actions(self, last_move_info=None):
        """
        外部调用的总闸门：获取当前所有的合法动作列表。
        last_move_info: 如果为空代表先手；否则代表要管的上家牌 (字典格式)
        """
        actions = []

        # === 场景 1：自由出牌 (先手或接风) ===
        if not last_move_info or last_move_info['type'] == TYPE_PASS:
            actions.extend(self.gen_singles())
            actions.extend(self.gen_pairs())
            actions.extend(self.gen_triples())
            actions.extend(self.gen_triple_pairs())
            actions.extend(self.gen_straights())
            actions.extend(self.gen_tubes())
            actions.extend(self.gen_plates())
            actions.extend(self.gen_straight_flushes())
            actions.extend(self.gen_bombs())
            actions.extend(self.gen_king_bomb())
            # 注意：先手绝对不能加入 PASS!
            return actions

        # === 场景 2：跟牌管牌 ===
        target_type = last_move_info['type']
        candidates = []

        # 1. 精准收集同类型的候选动作 (不越界计算)
        if target_type == TYPE_SINGLE:
            candidates = self.gen_singles()
        elif target_type == TYPE_PAIR:
            candidates = self.gen_pairs()
        elif target_type == TYPE_TRIPLE:
            candidates = self.gen_triples()
        elif target_type == TYPE_TRIPLE_PAIR:
            candidates = self.gen_triple_pairs()
        elif target_type == TYPE_STRAIGHT:
            candidates = self.gen_straights()
        elif target_type == TYPE_TUBE:
            candidates = self.gen_tubes()
        elif target_type == TYPE_PLATE:
            candidates = self.gen_plates()

        # 过滤出同类型中能压制的牌
        if candidates:
            actions.extend(self._filter_greater(candidates, last_move_info))

        # 2. 跨维打击：寻找能压制桌面的炸弹
        if target_type != TYPE_KING_BOMB:
            bomb_candidates = []
            bomb_candidates.extend(self.gen_straight_flushes())
            bomb_candidates.extend(self.gen_bombs())
            bomb_candidates.extend(self.gen_king_bomb())
            
            # 过滤出威力足够大的炸弹（比如能压住 5 炸的 6 炸或同花顺）
            actions.extend(self._filter_greater(bomb_candidates, last_move_info))

        # 3. 任何跟牌场景，永远可以选择 PASS (空列表)
        actions.append([]) 

        return actions


# ==========================================
# 测试脚本 (直接运行此文件)
# ==========================================
if __name__ == '__main__':
    def get_card_id(name, deck_num=1):
        """辅助函数：根据名称构造测试卡牌ID (deck_num=1或2 代表第一副或第二副牌)"""
        name = name.upper()
        if name in ['BJ', 'HR']: return 106 if deck_num == 1 else 107
        if name in ['SJ', 'SB']: return 104 if deck_num == 1 else 105
        
        for i in range(52):
            if get_card_name(i).upper() == name:
                return i if deck_num == 1 else i + 52
        return -1

    def parse_hand_input(input_str):
        """将用户输入的字符串（如 S3 S3 H2）转换为合法的卡牌ID列表"""
        input_str = input_str.replace(',', ' ')
        names = [x.strip().upper() for x in input_str.split() if x.strip()]
        
        from collections import Counter
        name_counts = Counter()
        hand_ids = []
        
        for name in names:
            name_counts[name] += 1
            deck_num = name_counts[name] # 记录这是该名称的第几张牌
            
            if deck_num > 2:
                print(f"警告：掼蛋最多只有两副牌，忽略多余的 {name}")
                continue
                
            cid = get_card_id(name, deck_num)
            if cid == -1:
                print(f"警告：无法识别的卡牌名称 '{name}'，已忽略")
                continue
                
            hand_ids.append(cid)
            
        return hand_ids

    # --- 1. 设定当前级牌 ---
    level_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
                 '10': 10, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    cur_level = 2
    while True:
        print("\n" + "="*40)
        lvl_input = input("请输入当前级牌 (2-A, 直接回车默认打2): ").strip().upper()
        if not lvl_input:
            break
        if lvl_input in level_map:
            cur_level = level_map[lvl_input]
            break
        print("输入无效，请输入 2, 3... 10, J, Q, K, A")
        
    cur_level_idx = cur_level - 2
    
    # --- 2. 手动输入手牌 ---
    while True:
        print("-" * 40)
        print("支持的牌名: S(黑桃) H(红心) C(梅花) D(方块) 加上 2~A")
        print("大小王输入: SB/SJ(小王), HR/BJ(大王)")
        print("输入示例: H2 H2 S3 S3 H3 C3 D4 D4 SB HR")
        hand_input = input("请输入你的手牌 (空格分隔): ").strip()
        
        if not hand_input:
            print("手牌不能为空，请重新输入")
            continue
            
        test_hand_ids = parse_hand_input(hand_input)
        if test_hand_ids:
            break
            
    # --- 3. 运行生成器 ---
    generator = MoveGenerator(test_hand_ids, cur_level_idx)
    
    print("\n" + "="*40)
    print(f"当前级牌: {cur_level}")
    print(f"当前手牌: {format_hand(test_hand_ids)} (共 {len(test_hand_ids)} 张)")
    print("="*40)
    
    # --- 4. 交互式测试：指定上家出牌 ---
    while True:
        print("\n" + "-"*40)
        print("请输入桌面上家出的牌 (空格分隔，如 'S8 C8' 或 'S3 S4 S5 S6 S7')")
        print("【直接回车】 -> 代表【先手出牌】(你是头家，随便出)")
        last_move_str = input("上家出牌: ").strip()

        last_move_info = None

        if last_move_str:
            last_move_ids = parse_hand_input(last_move_str)
            if not last_move_ids:
                continue
                
            # 调用 MoveDetector 解析这把牌是不是合法的，并获取特征字典
            last_move_info = MoveDetector.get_move_type(last_move_ids, cur_level_idx)
            
            if last_move_info is None:
                print(f"警告：你输入的 '{format_hand(last_move_ids)}' 不是合法的掼蛋牌型，请重新输入！")
                continue
                
            print(f"\n[场景: 桌面有牌 -> 牌型: {last_move_info['type']}, Rank: {last_move_info['rank']}, 张数: {last_move_info['count']}]")
        else:
            print("\n[场景: 先手出牌 (Free Play)]")

        # 获取合法动作
        actions = generator.get_legal_actions(last_move_info)

        # 打印输出
        print(f"-> 共计生成 {len(actions)} 种合法打法:\n")
        
        # 为了防止选项太多刷屏，我们分栏打印或截断显示
        for i, m in enumerate(actions):
            if not m:
                print(f"{i+1:3d}.  [PASS]")
            else:
                print(f"{i+1:3d}. ", format_hand(m))

        cont = input("\n 是否继续用当前手牌，测试其他的上家出牌？(y/n, 默认 y): ").strip().lower()
        if cont == 'n':
            break