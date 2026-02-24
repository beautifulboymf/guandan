import os
import sys
import glob
import torch
import random
import itertools
from collections import defaultdict

# 确保能正常导入 dmc 和 env
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dmc.models import GuandanModel
from dmc.env_wrapper import GuandanEnvWrapper

def play_match(model_A, model_B, device, num_games=4):
    """
    让 Model_A 和 Model_B 进行 2v2 对决。
    Model_A 固定坐在 0号和2号位（A队）；Model_B 固定坐在 1号和3号位（B队）。
    返回：A队的胜场数，B队的胜场数，A队的净胜分
    """
    env_wrapper = GuandanEnvWrapper()
    wins_A, wins_B = 0, 0
    score_A = 0.0
    
    for _ in range(num_games):
        # 随机抽取打几，确保全面考核
        obs = env_wrapper.reset(current_level=random.randint(2, 14))
        
        while True:
            cur_player = obs['player_id']
            legal_actions = obs['legal_actions']
            
            if not legal_actions: legal_actions = [[]]
            if len(legal_actions) == 1 and not legal_actions[0]:
                obs, reward, done, result_info = env_wrapper.step([])
                if done: break
                continue
            
            # 根据座位分配大脑
            active_model = model_A if cur_player in [0, 2] else model_B
            
            with torch.no_grad():
                scores = active_model(
                    torch.FloatTensor(obs['x_batch']['query']).to(device),
                    torch.FloatTensor(obs['x_batch']['context']).to(device),
                    torch.FloatTensor(obs['x_batch']['history']).to(device),
                    torch.FloatTensor(obs['x_batch']['history_mask']).to(device)
                ).squeeze(-1)
                
            best_action = legal_actions[torch.argmax(scores).item()]
            obs, reward, done, result_info = env_wrapper.step(best_action)
            
            if done: break
            
        # 统计结果
        result = result_info['result']
        team_A_won = (result['winner'] == 'A')
        level_up = float(result['level_up'])
        
        if team_A_won:
            wins_A += 1
            score_A += level_up
        else:
            wins_B += 1
            score_A -= level_up
            
    return wins_A, wins_B, score_A

def run_tournament():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚔️ AI 天下第一武道大会启动！运行设备: {device}")
    
    # 1. 收集所有参赛选手
    model_paths = glob.glob("history_models/*.pth")
    # 把当前的 best 和 latest 也加进参赛名单，看看它们是不是真的最强
    if os.path.exists("checkpoints/v4_selfplay_best.pth"):
        model_paths.append("checkpoints/v4_selfplay_best.pth")
    if os.path.exists("checkpoints/v4_selfplay_latest.pth"):
        model_paths.append("checkpoints/v4_selfplay_latest.pth")
        
    model_paths = list(set(model_paths)) # 去重
    
    if len(model_paths) < 2:
        print("❌ 参赛选手不足两人，请先积累更多历史模型！")
        return
        
    print(f"📜 共有 {len(model_paths)} 名选手参赛。")
    
    # 2. 内存复用优化：只在显存里驻留两个物理模型，不断替换灵魂（权重）
    model_A_container = GuandanModel(hidden_dim=512).to(device)
    model_B_container = GuandanModel(hidden_dim=512).to(device)
    model_A_container.eval()
    model_B_container.eval()
    
    # 3. 生成循环赛对阵表
    matchups = list(itertools.combinations(model_paths, 2))
    total_matches = len(matchups)
    
    # 每个对阵组合打 4 局（可以根据你想等多久适当调大）
    GAMES_PER_MATCH = 4 
    print(f"📅 赛程安排：共 {total_matches} 场对决，每场交锋 {GAMES_PER_MATCH} 局。预计总对局数：{total_matches * GAMES_PER_MATCH}")
    
    # 积分榜字典
    stats = defaultdict(lambda: {'wins': 0, 'games': 0, 'score_diff': 0.0})
    
    # 4. 开打
    for i, (path_A, path_B) in enumerate(matchups, 1):
        name_A = os.path.basename(path_A)
        name_B = os.path.basename(path_B)
        
        # 注入灵魂
        model_A_container.load_state_dict(torch.load(path_A, map_location=device, weights_only=True))
        model_B_container.load_state_dict(torch.load(path_B, map_location=device, weights_only=True))
        
        # 激战
        wins_A, wins_B, score_A = play_match(model_A_container, model_B_container, device, num_games=GAMES_PER_MATCH)
        
        # 记录战绩
        stats[path_A]['wins'] += wins_A
        stats[path_A]['games'] += GAMES_PER_MATCH
        stats[path_A]['score_diff'] += score_A
        
        stats[path_B]['wins'] += wins_B
        stats[path_B]['games'] += GAMES_PER_MATCH
        stats[path_B]['score_diff'] -= score_A
        
        # 打印赛况
        print(f"[{i}/{total_matches}] ⚔️ {name_A} (胜:{wins_A}) VS {name_B} (胜:{wins_B}) | A队净胜分: {score_A:+.1f}")
        
    # 5. 计算最终排行榜
    print("\n" + "="*50)
    print("🏆 掼蛋武林风云榜 (Leaderboard) 🏆")
    print("="*50)
    
    # 排序规则：第一参考 胜率，第二参考 净胜分
    leaderboard = []
    for path, data in stats.items():
        win_rate = data['wins'] / data['games']
        leaderboard.append((path, win_rate, data['score_diff'], data['wins'], data['games']))
        
    leaderboard.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    for rank, (path, win_rate, score_diff, wins, games) in enumerate(leaderboard, 1):
        name = os.path.basename(path)
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}"
        print(f"{medal} | 选手: {name:<25} | 胜率: {win_rate*100:5.1f}% | 净胜分: {score_diff:+6.1f} | 战绩: {wins}/{games}")
        
    print("="*50)
    
    # 6. 加冕为王
    champion_path = leaderboard[0][0]
    champion_save_path = "checkpoints/TOURNAMENT_CHAMPION.pth"
    
    import shutil
    shutil.copy(champion_path, champion_save_path)
    print(f"\n🎉 册封大典：已将最强模型 [{os.path.basename(champion_path)}] 复制并加冕为 -> {champion_save_path}")
    print("你可以直接拿着这个 TOURNAMENT_CHAMPION.pth 去和人类对战了！")

if __name__ == "__main__":
    run_tournament()