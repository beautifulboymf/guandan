import torch
import numpy as np
import random
import os
import sys

# 确保能找到项目根目录
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent

class Actor:
    """
    负责执行环境交互，收集一条完整的游戏轨迹 (Trajectory)。
    """
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon # 探索率 (Epsilon-Greedy)
        self.wrapper = GuandanEnvWrapper()
        
        # Player 0 是我们由神经网络控制的 AI
        # Player 1, 2, 3 是基于启发式规则的陪练 Bot
        self.bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}

    def play_episode(self):
        obs = self.wrapper.reset(current_level=2) # 默认从打 2 开始
        episode_data = [] 
        
        while True:
            cur_player = obs['player_id']
            legal_actions = obs['legal_actions']
            
            # 异常保护：如果没有任何动作，强制 PASS
            if not legal_actions:
                legal_actions = [[]]

            # 如果只有 PASS 一种选择，强制执行，不作为神经网络的决策样本
            if len(legal_actions) == 1 and not legal_actions[0]:
                obs, reward, done, result_info = self.wrapper.step([])
                if done: break
                continue

            # ==========================================
            # === AI (Player 0) 的决策回合 ===
            # ==========================================
            if cur_player == 0:
                x_batch = obs['x_batch']
                
                # 【核心修复】：动态获取当前模型所在的设备 (CPU 还是 CUDA)
                device = next(self.model.parameters()).device
                
                # 将 numpy 数组转换为 Tensor，并立刻推送到对应的设备上！
                query_t = torch.FloatTensor(x_batch['query']).to(device)
                context_t = torch.FloatTensor(x_batch['context']).to(device)
                history_t = torch.FloatTensor(x_batch['history']).to(device)
                mask_t = torch.FloatTensor(x_batch['history_mask']).to(device)
                
                # 神经网络打分 (停止记录梯度，极大加速推理)
                self.model.eval()
                with torch.no_grad():
                    scores = self.model(query_t, context_t, history_t, mask_t).squeeze(-1)
                
                # Epsilon-Greedy 探索策略
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, len(legal_actions) - 1)
                else:
                    action_idx = torch.argmax(scores).item()
                    
                best_action = legal_actions[action_idx]
                
                # 收集决策特征
                episode_data.append({
                    'query': x_batch['query'][action_idx],
                    'context': x_batch['context'][action_idx],
                    'history': x_batch['history'][action_idx],
                    'history_mask': x_batch['history_mask'][action_idx]
                })
                
                # 执行动作
                obs, reward, done, result_info = self.wrapper.step(best_action)
                
            # ==========================================
            # === 规则 Bot 的回合 (P1, P2, P3) ===
            # ==========================================
            else:
                action = self.bots[cur_player].act(obs['infoset'])
                obs, reward, done, result_info = self.wrapper.step(action)
                
            if done:
                break
                
        # ==========================================
        # === 高级 Reward Shaping 结算 ===
        # ==========================================
        result = result_info['result']
        winner_team = result['winner']      
        level_up = result['level_up']       
        
        if winner_team == 'A':
            team_score = float(level_up)    
        else:
            team_score = -float(level_up)   
            
        remaining_cards = len(self.wrapper.env.players_hand[0])
        penalty = remaining_cards * 0.05
        
        final_reward = team_score - penalty
        
        return episode_data, final_reward