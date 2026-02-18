import os
import sys
import time

# 确保能引用到 guandan 包
sys.path.append(os.getcwd())

# 修正引用路径，防止包路径混乱
try:
    from env.game import GameEnv
    from env.utils import format_hand, parse_input_string
except ImportError:
    from guandan.env.game import GameEnv
    from guandan.env.utils import format_hand, parse_input_string

def clear_screen():
    print("\n" * 3)
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
    print(header.center(60))
    print("-" * 60)
    
    tm_str = f"[对家 P{teammate}] 剩余: {info.others_num[teammate]} 张"
    print(tm_str.center(60))
    print("") 
    
    left_str = f"[上家 P{left}] 剩 {info.others_num[left]} 张"
    right_str = f"[下家 P{right}] 剩 {info.others_num[right]} 张"
    print(f"{left_str:<30}{right_str:>30}")
    
    print("-" * 60)
    
    if info.last_move:
        last_move_str = format_hand(info.last_move)
        last_pid_str = f"桌面最后出牌 (P{info.last_pid})"
        print(f"{last_pid_str}: {last_move_str}")
    else:
        print("桌面: [空] (请任意出牌)")
        
    print("-" * 60)
    print(f"玩家 P{me} 的手牌 - 共 {len(info.my_hand)} 张:")
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
        print("--- 游戏设置 ---")
        user_input = input("请输入当前级牌 (2-A, 默认2): ").strip().upper()
        
        if not user_input:
            return 2 # 默认打2
            
        if user_input in rank_map:
            return rank_map[user_input]
            
        print("输入无效，请输入 2, 3... 10, J, Q, K, A")
        time.sleep(1)

def main():
    # 1. 获取级牌设定
    start_level = get_start_level()
    
    env = GameEnv()
    # 2. 传入设定的级牌
    info = env.reset(current_level=start_level) 

    while not env.game_over:
        clear_screen()
        print_table(info)

        # 所有玩家由人类控制
        valid_input = False
        while not valid_input:
            prompt = f"[Player {info.player_id}] 请出牌 (空格分隔, 如 'H2 S2' 或 'pass'): "
            user_input = input(prompt).strip()
            
            if user_input.lower() == 'pass':
                action = []
                valid_input = True
            else:
                try:
                    action = parse_input_string(user_input, info.my_hand)
                    valid_input = True
                except ValueError as e:
                    print(f"输入错误: {e}")

        next_info, _, done, result = env.step(action)
        
        if 'error' in result:
            print(f"!!! 出牌无效: {result['error']} !!!")
            print("按回车键重试...")
            input()
            continue 
            
        info = next_info

    clear_screen()
    print("游戏结束!")
    if 'result' in result:
        print(f"结果: {result['result']['desc']}")
    print(f"出牌顺序: {env.winner_order}")

if __name__ == "__main__":
    main()