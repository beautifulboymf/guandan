import os
import sys
import time

# 确保能引用到 guandan 包
# 假设你在项目根目录下运行此脚本
sys.path.append(os.getcwd())

from env.game import GameEnv
from env.utils import format_hand, parse_input_string

def clear_screen():
    """简单的清屏效果"""
    print("\n" * 3)
    print("="*60)

def print_table(info):
    """可视化牌桌布局"""
    # 布局映射
    #      2 (Teammate/对家)
    # 1 (Prev/上家)      3 (Next/下家)
    #      0 (You/自己)
    
    me = info.player_id
    right = (me + 1) % 4
    teammate = (me + 2) % 4
    left = (me + 3) % 4
    
    # 1. 顶部状态栏
    header = f"当前级牌: {info.cur_level} | 当前视角: Player {me}"
    print(header.center(60))
    print("-" * 60)
    
    # 2. 对家 (居中)
    tm_str = f"[对家 P{teammate}] 剩余: {info.others_num[teammate]} 张"
    print(tm_str.center(60))
    print("") # 空一行增加视觉间隔
    
    # 3. 左右对手 (两侧对齐)
    left_str = f"[上家 P{left}] 剩 {info.others_num[left]} 张"
    right_str = f"[下家 P{right}] 剩 {info.others_num[right]} 张"
    # 使用 f-string 的填充功能进行左右对齐
    print(f"{left_str:<30}{right_str:>30}")
    
    print("-" * 60)
    
    # 4. 桌面出牌区
    if info.last_move:
        last_move_str = format_hand(info.last_move)
        last_pid_str = f"桌面最后出牌 (P{info.last_pid})"
        print(f"{last_pid_str}: {last_move_str}")
    else:
        print("桌面: [空] (请您的回合出牌)")
        
    print("-" * 60)
    
    # 5. 底部玩家手牌区
    print(f"我的手牌 (P{me}) - 共 {len(info.my_hand)} 张:")
    # 手牌已经被 format_hand 排序过 (大王>小王>级牌>普通牌)
    my_hand_str = format_hand(info.my_hand)
    print(my_hand_str)
    print("-" * 60)

def main():
    # 初始化环境
    env = GameEnv()
    # 假设当前打2
    info = env.reset(current_level=2) 

    while not env.game_over:
        clear_screen()
        print_table(info)

        # --- 玩家决策逻辑 (P0) ---
        if info.player_id == 0:
            valid_input = False
            while not valid_input:
                prompt = f"请输入要出的牌 (空格或逗号分隔, 如 'H2 S3' 或 'pass'): "
                user_input = input(prompt).strip()
                
                if user_input.lower() == 'pass':
                    action = []
                    valid_input = True
                else:
                    try:
                        # 解析输入的牌ID (仅检查手牌是否存在)
                        action = parse_input_string(user_input, info.my_hand)
                        valid_input = True
                    except ValueError as e:
                        print(f"输入错误: {e}")
        
        # --- AI (Dummy Agent) 逻辑 ---
        else:
            print(f"等待 Player {info.player_id} 出牌...")
            time.sleep(0.5) # 模拟思考时间
            
            # 简单的逻辑：
            # 如果桌上有牌且不是自己出的 -> PASS
            # 如果桌上没牌或者是自己出的 -> 出最小的一张
            if info.last_move and info.last_pid != info.player_id:
                action = [] # PASS
            else:
                if len(info.my_hand) > 0:
                    # 注意：MoveDetector会自动识别这是“单张”
                    action = [info.my_hand[0]]
                else:
                    action = []

        # --- 将动作提交给环境 ---
        # next_info: 新的状态
        # done: 游戏是否结束
        # result: 附加信息（包含错误提示）
        next_info, _, done, result = env.step(action)
        
        # --- 规则校验错误处理 ---
        if 'error' in result:
            print(f"!!! 出牌无效: {result['error']} !!!")
            print("按回车键重试...")
            input()
            # 遇到错误，跳过状态更新，直接进入下一次循环
            # 这样界面刷新后，还是轮到当前玩家出牌
            continue
            
        # 如果没有错误，更新当前 info，进入下一个玩家回合
        info = next_info

    # --- 游戏结束结算 ---
    clear_screen()
    print("游戏结束!")
    print(f"出牌顺序 (名次): {env.winner_order}")

if __name__ == "__main__":
    main()