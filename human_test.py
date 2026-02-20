import os
import sys
import time

# 确保能引用到 guandan 包
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.game import GameEnv
from env.utils import format_hand, parse_input_string
from dmc.agents import HeuristicAgent, RandomAgent

def clear_screen():
    # 打印空行模拟清屏，或者用 os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" * 2)
    print("="*60)

def print_table(info):
    me = info.player_id
    right = (me + 1) % 4
    teammate = (me + 2) % 4
    left = (me + 3) % 4
    
    # 将数字级牌转换为显示字符 (14->A, 10->T)
    level_str = str(info.cur_level)
    if info.cur_level == 10: level_str = '10'
    elif info.cur_level == 11: level_str = 'J'
    elif info.cur_level == 12: level_str = 'Q'
    elif info.cur_level == 13: level_str = 'K'
    elif info.cur_level == 14: level_str = 'A'
    
    header = f"当前级牌: {level_str} | 当前操作者: Player {me}"
    if me == 0:
        header += " (你)"
    print(header.center(60))
    print("-" * 60)
    
    tm_str = f"[对家 P{teammate}] 剩余: {info.others_num[teammate]} 张"
    print(tm_str.center(60))
    print("") 
    
    left_str = f"[上家 P{left}] 剩 {info.others_num.get(left, 0)} 张"
    right_str = f"[下家 P{right}] 剩 {info.others_num.get(right, 0)} 张"
    print(f"{left_str:<30}{right_str:>30}")
    
    print("-" * 60)
    
    if info.last_move:
        last_move_str = format_hand(info.last_move)
        last_pid_str = f"桌面最后出牌 (P{info.last_pid})"
        print(f"{last_pid_str}: {last_move_str}")
    else:
        print("桌面: [空] (请任意出牌)")
        
    print("-" * 60)
    if me == 0:
        print(f"你的手牌 - 共 {len(info.my_hand)} 张:")
        print(format_hand(info.my_hand))
        print("-" * 60)

def get_start_level():
    """获取初始级牌设置"""
    rank_map = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        '10': 10, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    
    while True:
        clear_screen()
        print("--- 掼蛋人机对战测试 ---")
        user_input = input("请输入当前级牌 (2-A, 默认打2): ").strip().upper()
        
        if not user_input:
            return 2 
            
        if user_input in rank_map:
            return rank_map[user_input]
            
        print("输入无效，请输入 2, 3... 10, J, Q, K, A")
        time.sleep(1)

def main():
    start_level = get_start_level()
    env = GameEnv()
    info = env.reset(current_level=start_level) 

    # 托管 P1, P2, P3 给规则 Bot
    agents = {
        1: HeuristicAgent(player_id=1),
        2: HeuristicAgent(player_id=2), # P2 是你的对家
        3: HeuristicAgent(player_id=3)
    }

    while not env.game_over:
        clear_screen()
        print_table(info)

        # === 人类玩家 (Player 0) 回合 ===
        if info.player_id == 0:
            valid_input = False
            while not valid_input:
                prompt = f"\n👉 [轮到你了] 请出牌 (空格分隔, 如 'H2 S2' 或 'pass'): "
                user_input = input(prompt).strip()
                
                if user_input.lower() == 'pass':
                    action = []
                    valid_input = True
                else:
                    try:
                        action = parse_input_string(user_input, info.my_hand)
                        valid_input = True
                    except ValueError as e:
                        print(f"❌ 输入错误: {e}")

            next_info, _, done, result = env.step(action)
            
            if 'error' in result:
                print(f"!!! 出牌无效: {result['error']} !!!")
                print("按回车键重试...")
                input()
                continue 
                
        # === 电脑玩家 (Player 1, 2, 3) 回合 ===
        else:
            # print(f"\n🤖 电脑 Player {info.player_id} 正在思考...")
            # time.sleep(1.0) # 停顿一下，让玩家有时间看清局势
            
            bot = agents[info.player_id]
            action = bot.act(info)
            
            action_str = format_hand(action) if action else "PASS"
            print(f"📣 Player {info.player_id} 出牌: {action_str}")
            # time.sleep(1.5) # 停顿一下，让玩家看清电脑出了什么
            
            next_info, _, done, result = env.step(action)
            
            # 如果电脑刚出完最后一张牌，提示一下
            if len(info.my_hand) - len(action) == 0:
                print(f"🎉 Player {info.player_id} 出完牌了！")
                # time.sleep(1)

        info = next_info

    # === 游戏结束结算 ===
    clear_screen()
    print("🎊 游戏结束! 🎊")
    if 'result' in result:
        print(f"对局结果: {result['result']['desc']}")
    print(f"出完牌的顺序: {env.winner_order}")
    
    # 判断你是赢了还是输了
    team_a = [0, 2]
    first_winner = env.winner_order[0]
    if first_winner in team_a:
        print("\n🏆 恭喜！你和 P2 的队伍获胜！")
    else:
        print("\n💔 遗憾！P1 和 P3 的队伍获胜！")

if __name__ == "__main__":
    main()