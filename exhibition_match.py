import os
import sys
import time
import torch
import random

# 确保能导入环境变量
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dmc.models import GuandanModel
from dmc.env_wrapper import GuandanEnvWrapper

# 终端输出颜色宏定义，增加观赏性
COLOR_A = "\033[96m"  # 青色 (A队: 0号, 2号)
COLOR_B = "\033[93m"  # 黄色 (B队: 1号, 3号)
COLOR_SYS = "\033[95m" # 紫色 (系统消息)
RESET = "\033[0m"

def get_card_str(card_id):
    """将单个数字 ID 解析为扑克牌字符串"""
    if card_id == 104 or card_id == 105:
        return "\033[90m🃏小王\033[0m"
    elif card_id == 106 or card_id == 107:
        return "\033[91m👑大王\033[0m"
    
    base_id = card_id % 52
    
    # ✅ 彻底修正：你的环境采用的是“按花色分块”的编码逻辑
    suit_idx = base_id // 13
    rank_idx = base_id % 13

    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['♠', '♥', '♣', '♦']
    
    suit_char = suits[suit_idx]
    rank_char = ranks[rank_idx]
    if suit_char in ['♥', '♦']:
        return f"\033[91m{suit_char}{rank_char}\033[0m" # 红桃方块高亮红色
    else:
        return f"{suit_char}{rank_char}"

def format_action(action):
    """将出牌动作格式化为华丽的扑克牌字符串"""
    if not action:
        return "Pass (过 ✋)"
    
    # 🧠【智能理牌优化】：按点数大小排序，而不是按ID乱排
    def sort_key(card_id):
        if card_id >= 104: return 999 + card_id # 大小王永远放在最右边
        rank = card_id % 13
        # 把 2 提到最后面 (如果你习惯 2 最大可以解开这行注释)
        # if rank == 0: rank = 14 
        return rank
        
    sorted_action = sorted(action, key=sort_key)
    card_strs = [get_card_str(cid) for cid in sorted_action]
    cards_display = " ".join(card_strs)
    
    return f"打出 {len(action):2d} 张牌 -> [{cards_display}]"

def run_exhibition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(COLOR_SYS + "==================================================" + RESET)
    print(COLOR_SYS + "      🏆 掼蛋 AI 巅峰表演赛 (Champion Showcase) 🏆" + RESET)
    print(COLOR_SYS + "==================================================" + RESET)
    print(f"🖥️ 正在将冠军灵魂注入 4 个克隆体... (设备: {device})")
    
    # 1. 加载冠军模型
    model_path = "checkpoints/TOURNAMENT_CHAMPION.pth"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件 {model_path}，请先运行天下第一武道大会！")
        return
        
    model = GuandanModel(hidden_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ 冠军模型加载完毕！四位神仙已就座。\n")

    # 2. 初始化环境
    env_wrapper = GuandanEnvWrapper()
    rand_level = random.randint(2, 14)
    level_map = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    level_str = level_map.get(rand_level, str(rand_level))
    
    print(COLOR_SYS + f"📢 裁判：本局游戏打【 {level_str} 】！红桃 {level_str} 为逢人配！" + RESET)
    print(COLOR_SYS + "📢 裁判：Game Start！\n" + RESET)
    
    obs = env_wrapper.reset(current_level=rand_level)

    # 3. 开始回合循环
    step = 1
    while True:
        cur_player = obs['player_id']
        legal_actions = obs['legal_actions']
        
        # 阵营颜色设定
        team_color = COLOR_A if cur_player in [0, 2] else COLOR_B
        team_name = "A队" if cur_player in [0, 2] else "B队"

        # 特殊情况处理：必须过牌
        if not legal_actions: legal_actions = [[]]
        if len(legal_actions) == 1 and not legal_actions[0]:
            print(team_color + f"[回合 {step:03d}] 玩家 {cur_player} ({team_name}): 被迫 Pass (过 ✋)" + RESET)
            obs, reward, done, result_info = env_wrapper.step([])
            if done: break
            step += 1
            # time.sleep(0.8) # 停顿 0.8 秒，模拟人类思考
            continue
        
        # 冠军模型进行神级推理
        with torch.no_grad():
            scores = model(
                torch.FloatTensor(obs['x_batch']['query']).to(device),
                torch.FloatTensor(obs['x_batch']['context']).to(device),
                torch.FloatTensor(obs['x_batch']['history']).to(device),
                torch.FloatTensor(obs['x_batch']['history_mask']).to(device)
            ).squeeze(-1)
            
        # 选择胜率最高的一招
        best_action = legal_actions[torch.argmax(scores).item()]
        
        # 打印出牌信息
        print(team_color + f"[回合 {step:03d}] 玩家 {cur_player} ({team_name}): {format_action(best_action)}" + RESET)

        # 环境推演下一步
        obs, reward, done, result_info = env_wrapper.step(best_action)
        if done: break
        
        step += 1
        # time.sleep(1.0) # 每次出牌停顿 1 秒，方便你看清局势

    # 4. 结算与战报
    print("\n" + COLOR_SYS + "==================================================" + RESET)
    print(COLOR_SYS + "                 🏁 游戏结束 🏁" + RESET)
    print(COLOR_SYS + "==================================================" + RESET)
    
    res = result_info['result']
    winner_team = res['winner']
    level_up = res['level_up']
    
    win_color = COLOR_A if winner_team == 'A' else COLOR_B
    print(win_color + f"🏆 获胜阵营: {winner_team} 队！" + RESET)
    print(win_color + f"⭐ 升级数: {level_up} 级" + RESET)
    print(COLOR_SYS + "详细对局数据:" + RESET)
    for k, v in res.items():
        print(f"  - {k}: {v}")
    print("\n")

if __name__ == '__main__':
    run_exhibition()