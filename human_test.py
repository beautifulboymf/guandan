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
    # 如果是在 guandan 目录下直接运行，尝试带包名的引用
    from env.game import GameEnv
    from env.utils import format_hand, parse_input_string

def clear_screen():
    print("\n" * 3)
    print("="*60)

def print_table(info):
    me = info.player_id
    right = (me + 1) % 4
    teammate = (me + 2) % 4
    left = (me + 3) % 4
    
    header = f"当前级牌: {info.cur_level} | 当前操作者: Player {me}"
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
        # 注意：这里我们无法直接获取上家的名字，只能通过 info.last_pid
        last_move_str = format_hand(info.last_move)
        last_pid_str = f"桌面最后出牌 (P{info.last_pid})"
        print(f"{last_pid_str}: {last_move_str}")
    else:
        print("桌面: [空] (请任意出牌)")
        
    print("-" * 60)
    print(f"玩家 P{me} 的手牌 - 共 {len(info.my_hand)} 张:")
    print(format_hand(info.my_hand))
    print("-" * 60)

def main():
    env = GameEnv()
    info = env.reset(current_level=2) 

    while not env.game_over:
        clear_screen()
        print_table(info)

        # === 修改点：所有玩家都由人类控制，方便测试规则 ===
        # 原来的逻辑是 if info.player_id == 0 else AI
        # 现在全部开放给输入
        
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

        # 提交动作
        next_info, _, done, result = env.step(action)
        
        # 错误检查
        if 'error' in result:
            print(f"!!! 出牌无效: {result['error']} !!!")
            print("按回车键重试...")
            input()
            continue # 重试，不更新 info
            
        info = next_info

    print("游戏结束!")
    print(f"出牌顺序: {env.winner_order}")

if __name__ == "__main__":
    main()