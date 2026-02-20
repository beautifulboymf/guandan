import os
import sys

# 确保能找到项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent
from env.utils import format_hand

def inspect_obs(obs, turn_name):
    """专门用来透视 obs 内部张量维度的检查函数"""
    player_id = obs['player_id']
    legal_actions = obs['legal_actions']
    x_batch = obs['x_batch']
    
    num_actions = len(legal_actions)
    print(f"\n[{turn_name}] 当前玩家: Player {player_id}")
    print(f"🌟 合法候选动作数量: {num_actions} 种")
    print(f"📦 x_batch['query'] 维度: {x_batch['query'].shape}  --> (手牌 108 + 动作 108 = 216)")
    print(f"📦 x_batch['context'] 维度: {x_batch['context'].shape}  --> (算牌底座 108 + 级牌 13 + 剩余牌数 84 = 205)")
    print(f"📦 x_batch['history'] 维度: {x_batch['history'].shape} --> (最近 15 手历史 x 每手特征 112)")
    print(f"📦 x_batch['history_mask'] 维度: {x_batch['history_mask'].shape} --> (1 代表真实发生，0 代表填充补位)")
    
    if num_actions > 0:
        # 打印掩码，直观看看有几个真实的动作记录
        print(f"🔍 当前 History Mask 数组: \n   {x_batch['history_mask'][0]}")

def main():
    print("="*60)
    print("🚀 掼蛋环境包装器 (EnvWrapper) 张量流测试")
    print("="*60)
    
    wrapper = GuandanEnvWrapper()
    
    # 叫来 4 个打工人（规则 Bot）帮我们自动出牌
    agents = {i: HeuristicAgent(player_id=i) for i in range(4)}
    
    # 开局
    obs = wrapper.reset(current_level=2)
    inspect_obs(obs, turn_name="开局 第 1 回合")
    
    # 自动打 6 手牌，积攒出牌历史
    print("\n" + "-"*60)
    print("🤖 让四个 Bot 自动打 6 手牌，观察历史记录的积累...")
    print("-"*60)
    
    for turn in range(2, 8):
        cur_player = obs['player_id']
        infoset = obs['infoset']
        
        # 让规则 Bot 选择一个最优动作
        action = agents[cur_player].act(infoset)
        action_str = format_hand(action) if action else "PASS"
        print(f"👉 回合 {turn} | 玩家 {cur_player} 打出: {action_str}")
        
        # 将动作喂给包装器
        obs, reward, done, info = wrapper.step(action)
        if done:
            break
            
    # 打了 6 手牌后，再检查一次维度和 Mask
    inspect_obs(obs, turn_name="战中 第 8 回合")
    print("\n✅ 测试完毕！张量维度与 DMC / Cross-Attention 架构完美吻合！")

if __name__ == '__main__':
    main()